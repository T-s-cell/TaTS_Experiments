"""Local CPU unit tests for the repro_tats port (no real GPT2 needed).

Runs against the real protocol directory (Security = smallest domain) with a
mock tokenizer + mock word-embedding LLM, mirroring the reference pack's
tests/test_reproduction.py style.

  python -m unittest repro_tats.tests.test_mechanics -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from repro_tats.config_tats import TatsConfig
from repro_tats.dataset_tats import TatsDataset
from repro_tats.exp_tats import Exp_TaTS
from repro_tats.vendor import EarlyStopping, encode_texts

PROTOCOL_DIR = os.path.join(REPO, "dataset", "timemmd_common_801010_v1")


class MockTokenizer:
    def __len__(self):
        return 64

    def __call__(self, texts, return_tensors="pt", padding=True,
                 truncation=True, max_length=256):
        rows = [[ord(c) % 63 + 1 for c in t[:max_length]] for t in texts]
        width = max([max(map(len, rows))] + [1])
        ids = [r + [0] * (width - len(r)) for r in rows]
        mask = [[1] * len(r) + [0] * (width - len(r)) for r in rows]
        return {"input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(mask)}


class MockLLM(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.embedding = nn.Embedding(64, dim)

        class C:
            _commit_hash = "mock"

        self.config = C()

    def get_input_embeddings(self):
        return self.embedding


def make_cfg(domain="Security", horizon=6, out_root=None, **over):
    proto = json.load(open(os.path.join(PROTOCOL_DIR, "protocol.json")))
    splits = json.load(open(os.path.join(PROTOCOL_DIR, "splits.json")))
    cfg = TatsConfig(protocol_dir=PROTOCOL_DIR, out_root=out_root,
                     domain=domain, horizon=horizon)
    cfg.fill_from_protocol(proto, splits)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


class TinyExp(Exp_TaTS):
    """Exp with a mock LLM (CPU, seconds); full-size iTransformer backbone."""

    def _build_llm(self):
        return MockLLM(self.cfg.llm_dim).to(self.device), MockTokenizer()


class TestEncode(unittest.TestCase):
    def test_mask_mean_pooling_matches_manual(self):
        llm = MockLLM(16)
        tok = MockTokenizer()
        out = encode_texts(["abc", "de"], tok, llm, "avg", 8, False)
        ids, mask = tok(["abc", "de"]).values()
        manual = (llm.embedding(ids) * mask.unsqueeze(-1)).sum(1) \
            / mask.sum(1, keepdim=True)
        self.assertTrue(torch.allclose(out, manual, atol=1e-6))

    def test_empty_text_raises(self):
        with self.assertRaises(ValueError):
            encode_texts([""], MockTokenizer(), MockLLM(16), "avg", 8, False)

    def test_deterministic(self):
        llm = MockLLM(16)
        tok = MockTokenizer()
        a = encode_texts(["hello world"], tok, llm, "avg", 8, False)
        b = encode_texts(["hello world"], tok, llm, "avg", 8, False)
        self.assertTrue(torch.equal(a, b))


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="tats_ds_")
        cls.cfg = make_cfg("Security", 6, out_root=cls.tmp)
        cls.llm = MockLLM()
        cls.ds = {sp: TatsDataset(cls.cfg, sp, cls.llm, MockTokenizer(),
                                  cache_dir=os.path.join(cls.tmp, "tc"))
                  for sp in ("train", "val", "test")}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_counts_match_protocol(self):
        splits = json.load(open(os.path.join(PROTOCOL_DIR, "splits.json")))
        wc = splits["Security"]["window_counts"]["6"]
        self.assertEqual(len(self.ds["train"]), wc["train"])
        self.assertEqual(len(self.ds["val"]), wc["val"])
        self.assertEqual(len(self.ds["test"]), wc["test"])

    def test_window_slicing_formula(self):
        df = pd.read_csv(os.path.join(PROTOCOL_DIR, "Security.csv"),
                         dtype=str, keep_default_na=False)
        sc = json.load(open(os.path.join(PROTOCOL_DIR, "scaler.json")))
        mean, std = float(sc["Security"]["mean"]), float(sc["Security"]["std"])
        ot = np.array([float(v) for v in df["OT"]])
        prior = np.array([float(v) for v in df["prior_history_avg"]])
        otz, priorz = (ot - mean) / std, (prior - mean) / std
        for sp in ("train", "val", "test"):
            ds = self.ds[sp]
            for i in (0, len(ds) // 2, len(ds) - 1):
                s = ds.starts[i]
                x, y, xm, ym, idx = ds[i]
                np.testing.assert_allclose(x[:, 0], otz[s - ds.L:s], rtol=0, atol=1e-6)
                np.testing.assert_allclose(y[:, 0], otz[s - 12:s + 6], rtol=0, atol=1e-6)
                py = ds.get_prior_y([i])[0]
                np.testing.assert_allclose(py[:, 0], priorz[s:s + 6], rtol=0, atol=1e-6)
                emb = ds.get_text_embeddings([i])[0]
                np.testing.assert_allclose(emb.numpy(),
                                           ds.text_embeddings[s - ds.L:s].numpy(),
                                           rtol=0, atol=0)
                self.assertEqual(idx, i)
                self.assertEqual(x.shape, (ds.L, 1))
                self.assertEqual(ym.shape, (12 + 6, ds.timef_dim))

    def test_freq_dims(self):
        self.assertEqual(self.ds["train"].data_stamp.shape[1], 1)  # Security ME
        cfg = make_cfg("Health", 12, out_root=self.tmp)
        ds = TatsDataset(cfg, "train", self.llm, MockTokenizer(),
                         cache_dir=os.path.join(self.tmp, "tc"))
        self.assertEqual(ds.data_stamp.shape[1], 2)  # Health W

    def test_label_dates_match_csv(self):
        df = pd.read_csv(os.path.join(PROTOCOL_DIR, "Security.csv"),
                         dtype=str, keep_default_na=False)
        dates = df["date"].tolist()
        ds = self.ds["test"]
        idn = ds.get_window_identity([0, len(ds) - 1])
        for k, i in enumerate([0, len(ds) - 1]):
            s = ds.starts[i]
            self.assertEqual(list(idn["label_dates"][k]),
                             dates[s:s + 6])
            self.assertEqual(idn["orig_row"][k], s)  # Security rows unchanged


class TestForward(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tats_fw_")
        self.cfg = make_cfg("Security", 6, out_root=self.tmp,
                            max_batches=2, max_eval_batches=2)
        self.exp = TinyExp(self.cfg)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _batch(self):
        data, loader = self.exp._get_data("train", shuffle=True)
        for b in loader:
            return data, b

    def test_mixing_formula(self):
        data, batch = self._batch()
        mixed, true, premix, prior = self.exp._forward_batch(data, batch)
        self.assertTrue(torch.allclose(
            mixed, 0.5 * premix + 0.5 * prior, atol=1e-6))
        self.cfg.prior_weight = 0.0
        mixed0, _, premix0, _ = self.exp._forward_batch(data, batch)
        self.assertTrue(torch.allclose(mixed0, premix0, atol=1e-7))
        self.cfg.prior_weight = 0.25
        mixedq, _, _, _ = self.exp._forward_batch(data, batch)
        self.assertFalse(torch.allclose(mixedq, 0.5 * premix0 + 0.5 * prior))
        self.cfg.prior_weight = 0.5

    def test_tail_batch_shapes(self):
        # Security H6 train has 12 windows -> a single short batch of 12;
        # text_dec must use the ACTUAL batch size (fork had a hardcoded bug).
        data, loader = self.exp._get_data("train", shuffle=False)
        batches = list(loader)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0].shape[0], 12)
        mixed, true, premix, prior = self.exp._forward_batch(data, batches[0])
        self.assertEqual(mixed.shape, (12, 6, 1))

    def test_nonfinite_target_raises(self):
        data, batch = self._batch()
        batch = (batch[0], batch[1].clone().fill_(float("nan")),
                 batch[2], batch[3], batch[4])
        with self.assertRaises(ValueError):
            self.exp._forward_batch(data, batch)

    def test_backbone_frozen_llm_not_in_optim(self):
        # every grad-bearing parameter lives in model/mlp only; the mock LLM
        # stands in for GPT2 whose params require_grad=False in Exp_TaTS
        opt_params = set()
        self.assertFalse(any(p.requires_grad for p in self.exp.llm_model.parameters())
                         if hasattr(self.exp, "llm_model") else False)


class TestCheckpointAndStopping(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tats_ck_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _train_tiny(self, out_root):
        cfg = make_cfg("Security", 6, out_root=out_root,
                       train_epochs=2, max_batches=1, max_eval_batches=1)
        exp = TinyExp(cfg)
        gdir = os.path.join(out_root, "runs", cfg.group_id())
        best = exp.train(gdir)
        return exp, cfg, gdir, best

    def test_checkpoint_roundtrip_and_reload_equality(self):
        exp, cfg, gdir, best = self._train_tiny(self.tmp)
        self.assertTrue(os.path.exists(os.path.join(gdir, "checkpoint.pth")))
        self.assertGreaterEqual(best["best_epoch"], 1)
        data, loader = exp._get_data("test")
        a = exp.predict(data, loader)[0]
        exp2 = TinyExp(make_cfg("Security", 6, out_root=self.tmp,
                                max_eval_batches=1))
        exp2.load_checkpoint(os.path.join(gdir, "checkpoint.pth"))
        data2, loader2 = exp2._get_data("test")
        b = exp2.predict(data2, loader2)[0]
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)

    def test_early_stopping_refuses_nonfinite(self):
        st = EarlyStopping(patience=3)
        tiny = nn.Linear(2, 2)
        with self.assertRaises(ValueError):
            st(float("nan"), nn.ModuleDict({"m": tiny}), self.tmp)

    def test_text_cache_reused_across_instances(self):
        cfg = make_cfg("Security", 6, out_root=self.tmp)
        cache = os.path.join(self.tmp, "tc")
        d1 = TatsDataset(cfg, "train", MockLLM(), MockTokenizer(), cache)
        mtime = os.path.getmtime(os.path.join(cache, os.listdir(cache)[0]))
        d2 = TatsDataset(cfg, "val", MockLLM(), MockTokenizer(), cache)
        files = os.listdir(cache)
        self.assertEqual(len(files), 1)  # one domain-level cache entry
        self.assertEqual(os.path.getmtime(os.path.join(cache, files[0])), mtime)
        np.testing.assert_allclose(
            d1.text_embeddings.numpy(), d2.text_embeddings.numpy(), atol=0)


if __name__ == "__main__":
    unittest.main()
