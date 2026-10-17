"""Protocol-driven TaTS dataset.

Windows come exclusively from the protocol manifest
(windows/<Domain>_H<h>.csv); no re-splitting happens here. `s` in the
manifest is the global label-start row: input rows [s-L, s), label rows
[s, s+H). Text embeddings are encoded once per domain through the shared
encode_texts path and cached on disk, so train/val/test (and every later
group of the same domain) read identical embeddings.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.timefeatures import time_features
from repro_tats.vendor import encode_texts


def _sha(texts_bytes):
    return hashlib.sha256(texts_bytes).hexdigest()


def load_or_encode_texts(protocol_dir, domain, texts, tokenizer, llm_model,
                         cfg, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    key_src = json.dumps({
        "domain": domain,
        "texts_sha256": _sha("\x1e".join(texts).encode("utf-8")),
        "tokenizer": type(tokenizer).__name__,
        "vocab": len(tokenizer),
        "llm_commit": getattr(getattr(llm_model, "config", None),
                              "_commit_hash", None),
        "pool_type": cfg.pool_type,
        "use_fullmodel": cfg.use_fullmodel,
        "max_length": cfg.text_max_length,
        "batch_size": cfg.text_batch_size,
    }, sort_keys=True)
    key = hashlib.sha256(key_src.encode()).hexdigest()[:32]
    path = os.path.join(cache_dir, "textemb_%s_%s.npy" % (domain, key))
    if os.path.exists(path):
        arr = np.load(path)
        emb = torch.from_numpy(arr)
        return emb.to(next(llm_model.parameters()).device)
    emb = encode_texts(texts, tokenizer, llm_model, cfg.pool_type,
                       cfg.text_batch_size, bool(cfg.use_fullmodel))
    np.save(path, emb.cpu().numpy().astype(np.float32))
    return emb


class TatsDataset(Dataset):
    def __init__(self, cfg, split, llm_model, tokenizer, cache_dir):
        assert split in ("train", "val", "test")
        self.cfg = cfg
        self.split = split
        pdir = cfg.protocol_dir
        proto = json.load(open(os.path.join(pdir, "protocol.json")))
        d = proto["domains"][cfg.domain]
        self.field_map = d["field_map"]
        self.L = d["L"]
        self.H = cfg.horizon
        self.timef_dim = d["timef_dim"]

        df = pd.read_csv(os.path.join(pdir, cfg.domain + ".csv"),
                         dtype=str, keep_default_na=False)
        n = len(df)
        scaler = json.load(open(os.path.join(pdir, "scaler.json")))[cfg.domain]
        mean, std = float(scaler["mean"]), float(scaler["std"])

        ot = np.array([float(v) for v in df["OT"]], dtype=np.float64)
        prior = np.array([float(v) for v in df[self.field_map["prior"]]],
                         dtype=np.float64)
        self.data_x = ((ot - mean) / std).astype(np.float32)[:, None]
        self.data_prior = ((prior - mean) / std).astype(np.float32)[:, None]

        dates = pd.to_datetime(df[self.field_map["date"]],
                               format="%Y-%m-%d", errors="raise")
        stamp = time_features(pd.DatetimeIndex(dates), freq=cfg.freq)
        self.data_stamp = stamp.transpose(1, 0).astype(np.float32)  # (N, timef)
        self.label_dates_iso = np.array(
            [d_.strftime("%Y-%m-%d") for d_ in dates], dtype="U10")

        w = pd.read_csv(os.path.join(
            pdir, "windows", "%s_H%d.csv" % (cfg.domain, cfg.horizon)))
        rows = w[w["split"] == split].sort_values("s")
        self.starts = rows["s"].astype(int).tolist()
        self.orig_rows = rows["orig_row"].astype(int).tolist()
        if not self.starts:
            raise ValueError("empty split %s for %s H%d"
                             % (split, cfg.domain, cfg.horizon))

        texts = df[self.field_map["fact"]].tolist()
        self.text_embeddings = load_or_encode_texts(
            pdir, cfg.domain, texts, tokenizer, llm_model, cfg, cache_dir)

    def __len__(self):
        return len(self.starts)

    def _rows_for(self, indices):
        """Map dataset window indices -> global label-start rows."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return [self.starts[i] for i in indices]

    def get_prior_y(self, indices):
        return np.stack([self.data_prior[s:s + self.H]
                         for s in self._rows_for(indices)])

    def get_text_embeddings(self, indices):
        return torch.stack([self.text_embeddings[s - self.L:s]
                            for s in self._rows_for(indices)])

    def get_window_identity(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return {
            "s": np.array([self.starts[i] for i in indices], dtype=np.int64),
            "orig_row": np.array([self.orig_rows[i] for i in indices],
                                 dtype=np.int64),
            "label_dates": np.stack(
                [self.label_dates_iso[self.starts[i]:
                                      self.starts[i] + self.H]
                 for i in indices]),
        }

    def __getitem__(self, index):
        s = self.starts[index]
        L, lab = self.L, self.cfg.label_len
        seq_x = self.data_x[s - L:s]
        seq_y = self.data_y_slice(s, lab)
        x_mark = self.data_stamp[s - L:s]
        y_mark = self.data_stamp[s - lab:s + self.H]
        return seq_x, seq_y, x_mark, y_mark, index

    def data_y_slice(self, s, lab):
        r_begin = s - lab
        r_end = s + self.H
        return self.data_x[r_begin:r_end]
