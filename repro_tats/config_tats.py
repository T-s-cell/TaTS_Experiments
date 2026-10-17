"""Explicit configuration for one TaTS reproduction group (domain, H)."""
import json
import os
from dataclasses import dataclass, field, asdict


@dataclass
class TatsConfig:
    protocol_dir: str
    out_root: str
    domain: str
    horizon: int
    seed: int = 2021

    # model: iTransformer + frozen GPT2 word embeddings
    task_name: str = "long_term_forecast"
    model: str = "iTransformer"
    llm_model: str = "GPT2"
    llm_dim: int = 768
    llm_layers: int = 6
    use_fullmodel: int = 0
    text_emb: int = 12
    pool_type: str = "avg"
    text_max_length: int = 256
    text_batch_size: int = 64
    prior_weight: float = 0.5

    # iTransformer backbone (fork run.py defaults, per corrected reproduction)
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 2048
    factor: int = 1
    dropout: float = 0.1
    activation: str = "gelu"
    embed: str = "timeF"
    output_attention: bool = False
    use_norm: int = 1

    # sequence
    label_len: int = 12

    # optimization
    batch_size: int = 32
    train_epochs: int = 50
    patience: int = 20
    learning_rate: float = 1e-4
    learning_rate2: float = 1e-2
    num_workers: int = 0

    # reproducibility
    use_amp: bool = False
    allow_tf32: bool = False

    # preflight caps (0 = unlimited)
    max_batches: int = 0
    max_eval_batches: int = 0

    # filled from protocol
    seq_len: int = 0
    freq: str = ""
    n_train: int = 0
    val_end: int = 0
    n: int = 0

    def fill_from_protocol(self, protocol: dict, splits: dict):
        d = protocol["domains"][self.domain]
        s = splits[self.domain]
        if self.horizon not in d["H"]:
            raise ValueError("horizon %d not in %s" % (self.horizon, d["H"]))
        self.seq_len = d["L"]
        self.freq = d["freq"]
        self.n = s["N"]
        self.n_train = s["n_train"]
        self.val_end = s["val_end"]

    def group_id(self):
        return "%s_H%d_seed%d" % (self.domain, self.horizon, self.seed)

    @property
    def pred_len(self):
        return self.horizon

    def public_dict(self):
        return {k: v for k, v in asdict(self).items()}
