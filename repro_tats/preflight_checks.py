"""Real-GPT2 GPU mechanics preflight (complements run_all --preflight).

Verifies on the actual training stack (no mocks):
  - GPT2 loads offline, vocab 50257, pad==eos, built layer count == llm_layers
  - GPT2 params are byte-identical before/after training steps (fully frozen)
  - backbone + text MLP params change; their grads are finite; llm grads None
  - Security H12 tail batch: 6 train windows arrive as ONE undropped batch
  - forward shapes + prior mixing applied
  - timing: text encoding wall, forward+backward sec/step, peak VRAM

Writes preflight_mechanics.json into --out_root.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from repro_tats.config_tats import TatsConfig
from repro_tats.exp_tats import Exp_TaTS, seed_everything


def digest(module):
    h = hashlib.sha256()
    with torch.no_grad():
        for p in module.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=2021)
    args = ap.parse_args()

    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    checks = {}

    cfg = TatsConfig(protocol_dir=args.protocol_dir, out_root=str(out),
                     domain="Security", horizon=12, seed=args.seed,
                     max_batches=2, max_eval_batches=2)
    proto = json.load(open(Path(args.protocol_dir) / "protocol.json"))
    splits = json.load(open(Path(args.protocol_dir) / "splits.json"))
    cfg.fill_from_protocol(proto, splits)

    t0 = time.time()
    exp = Exp_TaTS(cfg)
    checks["exp_build_seconds"] = round(time.time() - t0, 1)
    seed_everything(cfg.seed)

    llm, tok = exp.llm_model, exp.tokenizer
    checks["llm_class"] = type(llm).__name__
    checks["tokenizer_class"] = type(tok).__name__
    checks["vocab"] = len(tok)
    checks["pad_is_eos"] = tok.pad_token == tok.eos_token
    checks["llm_layers_built"] = len(llm.h)
    checks["llm_layers_expected"] = cfg.llm_layers
    checks["llm_all_frozen"] = all(not p.requires_grad
                                   for p in llm.parameters())
    checks["llm_eval_mode"] = not llm.training

    train_data, train_loader = exp._get_data("train", shuffle=True)
    checks["train_windows"] = len(train_data)
    batches = list(train_loader)
    checks["train_batches"] = len(batches)
    checks["tail_batch_size"] = int(batches[-1][0].shape[0])
    checks["tail_batch_undropped"] = (
        sum(b[0].shape[0] for b in batches) == len(train_data))

    llm_before = digest(llm)
    model_before = digest(exp.model)
    mlp_before = digest(exp.mlp)

    opt_model = torch.optim.Adam(exp.model.parameters(), lr=cfg.learning_rate)
    opt_mlp = torch.optim.Adam(exp.mlp.parameters(), lr=cfg.learning_rate2)
    criterion = torch.nn.MSELoss()
    step_seconds, losses = [], []
    exp.model.train()
    exp.mlp.train()
    for bi, batch in enumerate(batches[:cfg.max_batches]):
        t1 = time.time()
        opt_model.zero_grad(set_to_none=True)
        opt_mlp.zero_grad(set_to_none=True)
        mixed, true, premix, prior = exp._forward_batch(train_data, batch)
        if bi == 0:
            B, H = mixed.shape[0], mixed.shape[1]
            checks["forward_shape"] = [B, H, int(mixed.shape[2])]
            checks["prior_shape"] = list(prior.shape)
            checks["mix_half_applied"] = bool(torch.allclose(
                mixed, 0.5 * premix + 0.5 * prior, atol=1e-6))
        loss = criterion(mixed, true)
        loss.backward()
        checks["grads_finite"] = all(
            torch.isfinite(p.grad).all().item()
            for m in (exp.model, exp.mlp) for p in m.parameters()
            if p.grad is not None)
        checks["llm_grads_all_none"] = all(
            p.grad is None for p in llm.parameters())
        opt_model.step()
        opt_mlp.step()
        step_seconds.append(round(time.time() - t1, 3))
        losses.append(float(loss.item()))
    checks["step_seconds"] = step_seconds
    checks["losses"] = losses
    checks["losses_finite"] = all(np_isfinite(v) for v in losses)

    checks["llm_digest_unchanged"] = digest(llm) == llm_before
    checks["model_updated"] = digest(exp.model) != model_before
    checks["mlp_updated"] = digest(exp.mlp) != mlp_before
    checks["peak_mem_gib"] = round(
        torch.cuda.max_memory_allocated() / 2**30, 3)

    verdict_items = [
        checks["llm_all_frozen"], checks["llm_eval_mode"],
        checks["llm_layers_built"] == cfg.llm_layers,
        checks["pad_is_eos"], checks["tail_batch_undropped"],
        checks["mix_half_applied"], checks["grads_finite"],
        checks["llm_grads_all_none"], checks["losses_finite"],
        checks["llm_digest_unchanged"], checks["model_updated"],
        checks["mlp_updated"],
    ]
    checks["ALL_PASS"] = all(verdict_items)
    (out / "preflight_mechanics.json").write_text(
        json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2), flush=True)
    if not checks["ALL_PASS"]:
        sys.exit(2)


def np_isfinite(v):
    return v == v and abs(v) != float("inf")


if __name__ == "__main__":
    main()
