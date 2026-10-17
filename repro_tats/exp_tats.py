"""TaTS experiment: iTransformer backbone + frozen GPT2 text embeddings.

Ported mechanisms from the validated reference implementation:
  - encode via word-embedding lookup (use_fullmodel=0), attention-mask mean
    pooling, identical for train/val/test (dataset-level shared cache)
  - forward mixes the prior: pred = (1-pw)*model_output + pw*prior_y
  - full-element validation MSE (concatenate all windows, weight the short
    tail batch by its true element count)
  - validation-only best-epoch selection; EarlyStopping refuses non-finite
  - checkpoints hold BOTH the backbone and the text MLP (ModuleDict, strict)
  - non-finite forward outputs / training loss raise immediately
  - GPT2 fully frozen (requires_grad=False, eval, no_grad), pad_token=eos
"""
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from models.iTransformer import Model as ITransformer
from repro_tats.vendor import EarlyStopping, metric

from repro_tats.dataset_tats import TatsDataset


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strict_matmul_flags():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


class Exp_TaTS:
    def __init__(self, cfg):
        self.cfg = cfg
        strict_matmul_flags()
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")
        seed_everything(cfg.seed)

        self.model = self._build_backbone().to(self.device)
        self.llm_model, self.tokenizer = self._build_llm()
        for p in self.llm_model.parameters():
            p.requires_grad = False
        self.llm_model.eval()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.llm_dim, cfg.llm_dim // 8),
            nn.ReLU(),
            nn.Linear(cfg.llm_dim // 8, cfg.text_emb),
            nn.ReLU(),
            nn.Dropout(0.3),
        ).to(self.device)
        n_ts = sum(p.numel() for p in self.model.parameters())
        n_mlp = sum(p.numel() for p in self.mlp.parameters())
        print("params: TS model %d, text MLP %d, total %d"
              % (n_ts, n_mlp, n_ts + n_mlp), flush=True)
        self._data_cache = {}

    # ---------- construction ----------
    def _build_backbone(self):
        return ITransformer(self.cfg).float()

    def _build_llm(self):
        from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
        gpt2_config = GPT2Config.from_pretrained(
            "openai-community/gpt2", local_files_only=True)
        gpt2_config.num_hidden_layers = self.cfg.llm_layers
        try:
            llm = GPT2Model.from_pretrained(
                "openai-community/gpt2", local_files_only=True,
                config=gpt2_config)
            tokenizer = GPT2Tokenizer.from_pretrained(
                "openai-community/gpt2", local_files_only=True)
        except EnvironmentError:
            print("Local GPT2 files not found; attempting download", flush=True)
            llm = GPT2Model.from_pretrained(
                "openai-community/gpt2", local_files_only=False,
                config=gpt2_config)
            tokenizer = GPT2Tokenizer.from_pretrained(
                "openai-community/gpt2", local_files_only=False)
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.pad_token = "[PAD]"
        llm = llm.to(self.device)
        return llm, tokenizer

    def _get_data(self, split, shuffle=False):
        key = (split, shuffle)
        if key not in self._data_cache:
            ds = TatsDataset(self.cfg, split, self.llm_model, self.tokenizer,
                             cache_dir=os.path.join(self.cfg.out_root,
                                                    "text_cache"))
            g = torch.Generator()
            g.manual_seed(self.cfg.seed)
            loader = DataLoader(ds, batch_size=self.cfg.batch_size,
                                shuffle=shuffle, drop_last=False,
                                num_workers=self.cfg.num_workers,
                                generator=g if shuffle else None)
            self._data_cache[key] = (ds, loader)
        return self._data_cache[key]

    def _checkpoint_modules(self):
        return nn.ModuleDict({"model": self.model, "mlp": self.mlp})

    # ---------- forward ----------
    def _forward_batch(self, data, batch):
        batch_x, batch_y, x_mark, y_mark, indices = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        x_mark = x_mark.float().to(self.device)
        y_mark = y_mark.float().to(self.device)
        prior_y = torch.as_tensor(data.get_prior_y(indices),
                                  dtype=torch.float32, device=self.device)
        prompt = self.mlp(data.get_text_embeddings(indices))
        B = batch_x.shape[0]
        dec = torch.cat([batch_y[:, :self.cfg.label_len],
                         batch_y.new_zeros(B, self.cfg.horizon, 1)], dim=1)
        text_dec = torch.cat([prompt[:, -self.cfg.label_len:],
                              prompt.new_zeros(B, self.cfg.horizon,
                                               self.cfg.text_emb)], dim=1)
        outputs = self.model(torch.cat([batch_x, prompt], dim=-1), x_mark,
                             torch.cat([dec, text_dec], dim=-1), y_mark)
        outputs = outputs[:, -self.cfg.horizon:, :1]
        premix = outputs
        pw = self.cfg.prior_weight
        mixed = (1.0 - pw) * outputs + pw * prior_y
        target = batch_y[:, -self.cfg.horizon:, :1]
        if not (torch.isfinite(outputs).all() and torch.isfinite(target).all()
                and torch.isfinite(mixed).all()):
            raise ValueError("Non-finite forecast or target")
        return mixed, target, premix, prior_y

    def predict(self, data, loader):
        self.model.eval()
        self.mlp.eval()
        mixed_l, premix_l, prior_l, true_l, idx_l = [], [], [], [], []
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if self.cfg.max_eval_batches and bi >= self.cfg.max_eval_batches:
                    break
                mixed, true, premix, prior = self._forward_batch(data, batch)
                mixed_l.append(mixed.float().cpu().numpy())
                premix_l.append(premix.float().cpu().numpy())
                prior_l.append(prior.float().cpu().numpy())
                true_l.append(true.float().cpu().numpy())
                idx_l.append(batch[4].numpy())
        if not mixed_l:
            raise ValueError("evaluation loader is empty")
        out = tuple(np.concatenate(a, axis=0) for a in
                    (mixed_l, premix_l, prior_l, true_l, idx_l))
        if not self.cfg.max_eval_batches and len(out[0]) != len(data):
            raise ValueError("evaluation did not cover every window")
        return out

    def vali(self, data, loader):
        mixed, _, _, true, _ = self.predict(data, loader)
        mae, mse = metric(mixed, true)[:2]
        self.model.train()
        self.mlp.train()
        return float(mse), float(mae)

    # ---------- checkpoint io ----------
    def save_checkpoint(self, path):
        torch.save(self._checkpoint_modules().state_dict(), path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if not any(k.startswith("mlp.") for k in state):
            raise ValueError("checkpoint lacks the trained text MLP")
        self._checkpoint_modules().load_state_dict(state, strict=True)

    # ---------- manifest ----------
    def manifest(self, extra=None):
        cfg = self.cfg
        pdir = Path(cfg.protocol_dir)
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        files = ["%s.csv" % cfg.domain,
                 "windows/%s_H%d.csv" % (cfg.domain, cfg.horizon),
                 "splits.json", "scaler.json", "protocol.json"]
        man = {
            "config": cfg.public_dict(),
            "protocol_sha256": {f: digest(pdir / f) for f in files},
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve()
                .parents[1], capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda" else "cpu",
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_vocab_size": len(self.tokenizer),
            "llm_commit": getattr(self.llm_model.config, "_commit_hash", None),
        }
        if extra:
            man.update(extra)
        return man

    # ---------- training ----------
    def train(self, group_dir):
        cfg = self.cfg
        train_data, train_loader = self._get_data("train", shuffle=True)
        vali_data, vali_loader = self._get_data("val")
        gdir = Path(group_dir)
        gdir.mkdir(parents=True, exist_ok=True)
        ckpt = gdir / "checkpoint.pth"
        if ckpt.exists():
            raise FileExistsError("refusing to overwrite %s" % ckpt)

        opt_model = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        opt_mlp = optim.Adam(self.mlp.parameters(), lr=cfg.learning_rate2)
        criterion = nn.MSELoss()
        stopper = EarlyStopping(patience=cfg.patience, verbose=True)
        modules = self._checkpoint_modules()

        history, best_epoch = [], None
        t0 = time.time()
        for epoch in range(cfg.train_epochs):
            te = time.time()
            self.model.train()
            self.mlp.train()
            loss_sum, count = 0.0, 0
            for bi, batch in enumerate(train_loader):
                if cfg.max_batches and bi >= cfg.max_batches:
                    break
                opt_model.zero_grad(set_to_none=True)
                opt_mlp.zero_grad(set_to_none=True)
                mixed, true, _, _ = self._forward_batch(train_data, batch)
                loss = criterion(mixed, true)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                loss.backward()
                opt_model.step()
                opt_mlp.step()
                loss_sum += loss.item() * true.numel()
                count += true.numel()
            if not count:
                raise ValueError("train loader is empty")
            val_mse, val_mae = self.vali(vali_data, vali_loader)
            improved = val_mse < stopper.val_loss_min
            stopper(val_mse, modules, str(gdir))
            if improved:
                best_epoch = epoch + 1
            rec = {"epoch": epoch + 1,
                   "train_mse": loss_sum / count,
                   "val_mse": val_mse, "val_mae": val_mae,
                   "seconds": round(time.time() - te, 3)}
            history.append(rec)
            print("Epoch: " + json.dumps(rec), flush=True)
            if stopper.early_stop:
                print("Early stopping", flush=True)
                break
        wall = time.time() - t0
        best = {"best_epoch": best_epoch,
                "best_val_mse": float(stopper.val_loss_min),
                "epochs_run": len(history),
                "train_seconds": round(wall, 1),
                "training_windows": len(train_data),
                "validation_windows": len(vali_data)}
        man = self.manifest(best)
        (gdir / "manifest.json").write_text(
            json.dumps(man, indent=2, ensure_ascii=False))
        (gdir / "history.json").write_text(json.dumps(history, indent=2))
        self.load_checkpoint(str(ckpt))
        return best
