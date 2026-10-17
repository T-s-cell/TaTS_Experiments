"""Run one TaTS reproduction group end-to-end.

Modes:
  train       train -> best-checkpoint test -> save npz/metrics, then spawn a
              NEW process (--mode test_only) that reloads the checkpoint and
              verifies predictions match within 1e-6, plus float64 recompute.
  test_only   independent reload verification (no training): checkpoint ->
              predict -> compare against train-phase npz -> reload_check.json
              + recompute_metrics.json (float64).

Gates (exit 2): any reload diff > 1e-6, non-natural stop, missing artifacts.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repro_tats.config_tats import TatsConfig
from repro_tats.exp_tats import Exp_TaTS, seed_everything

RELOAD_TOL = 1e-6


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cfg(args):
    pdir = args.protocol_dir
    proto = json.load(open(os.path.join(pdir, "protocol.json")))
    splits = json.load(open(os.path.join(pdir, "splits.json")))
    for rel, want in proto["files_sha256"].items():
        got = sha256(os.path.join(pdir, rel))
        if got != want:
            raise SystemExit("[FATAL] protocol file %s sha mismatch" % rel)
    cfg = TatsConfig(protocol_dir=pdir, out_root=args.out_root,
                     domain=args.domain, horizon=args.horizon,
                     seed=args.seed, max_batches=args.max_batches,
                     max_eval_batches=args.max_eval_batches)
    cfg.fill_from_protocol(proto, splits)
    return cfg


def save_test_arrays(gdir, arrays, identity):
    np.savez_compressed(
        os.path.join(gdir, "test_arrays.npz"),
        pred_mixed=arrays[0][:, :, 0], pred_model=arrays[1][:, :, 0],
        prior_y=arrays[2][:, :, 0], truth=arrays[3][:, :, 0],
        s=identity["s"], orig_row=identity["orig_row"],
        label_dates=identity["label_dates"], window_index=arrays[4])


def metrics_f32(pred, true):
    diff = pred - true
    mse = float(np.mean(diff.astype(np.float32) ** 2))
    mae = float(np.mean(np.abs(diff.astype(np.float32))))
    per_step = []
    for k in range(pred.shape[1]):
        d = diff[:, k]
        per_step.append({"step": k + 1,
                         "mse": float(np.mean((d ** 2).astype(np.float32))),
                         "mae": float(np.mean(np.abs(d).astype(np.float32)))})
    return mse, mae, per_step


def metrics_f64(pred, true):
    p = pred.astype(np.float64)
    t = true.astype(np.float64)
    d = p - t
    return float(np.mean(d ** 2)), float(np.mean(np.abs(d)))


def do_test_phase(exp, gdir, tag):
    data, loader = exp._get_data("test")
    arrays = exp.predict(data, loader)
    idx = arrays[4].tolist()
    identity = data.get_window_identity(idx)
    save_test_arrays(gdir, arrays, identity)
    mse, mae, per_step = metrics_f32(arrays[0][:, :, 0], arrays[3][:, :, 0])
    if not all(np.isfinite(v) for v in (mse, mae)):
        raise ValueError("non-finite test metrics")
    result = {"tag": tag, "mse": mse, "mae": mae,
              "evaluated_windows": len(arrays[0]),
              "expected_windows": len(data),
              "per_step": per_step}
    if len(arrays[0]) != len(data):
        raise ValueError("test coverage %d != %d" % (len(arrays[0]), len(data)))
    (gdir / ("test_metrics_%s.json" % tag)).write_text(
        json.dumps(result, indent=2))
    return arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "test_only"], required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--protocol_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--max_eval_batches", type=int, default=0)
    ap.add_argument("--reload_tol", type=float, default=RELOAD_TOL)
    args = ap.parse_args()

    cfg = build_cfg(args)
    gid = cfg.group_id()
    gdir = Path(cfg.out_root) / "runs" / gid
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "run_config.json").write_text(
        json.dumps(cfg.public_dict(), indent=2))
    ckpt = gdir / "checkpoint.pth"

    seed_everything(cfg.seed)
    exp = Exp_TaTS(cfg)

    if args.mode == "train":
        if ckpt.exists():
            raise SystemExit("[FATAL] refusing to overwrite %s" % ckpt)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        best = exp.train(str(gdir))
        train_wall = time.time() - t0
        arrays = do_test_phase(exp, gdir, "after_train")
        peak = torch.cuda.max_memory_allocated() / 2**30 \
            if torch.cuda.is_available() else 0.0
        (gdir / "train_phase.json").write_text(json.dumps(
            {**best, "test_mse": None, "peak_mem_gib": round(peak, 3),
             "train_wall_seconds": round(train_wall, 1),
             "protocol": os.path.abspath(cfg.protocol_dir),
             "out_root": os.path.abspath(cfg.out_root)}, indent=2))
        # independent-process reload check + float64 recompute
        env = dict(os.environ)
        r = subprocess.run(
            [sys.executable, "-u", Path(__file__).resolve(),
             "--mode", "test_only", "--domain", args.domain,
             "--horizon", str(args.horizon),
             "--protocol_dir", args.protocol_dir,
             "--out_root", args.out_root, "--seed", str(args.seed),
             "--max_eval_batches", str(args.max_eval_batches)],
            env=env, cwd=str(Path(__file__).resolve().parents[1]))
        if r.returncode != 0:
            raise SystemExit("[FATAL] reload/recompute phase failed rc=%d" % r.returncode)
        print("GROUP %s DONE" % gid, flush=True)

    else:  # test_only: fresh process, checkpoint reload verification
        exp.load_checkpoint(str(ckpt))
        arrays = do_test_phase(exp, gdir, "reload")
        ref = np.load(gdir / "test_arrays.npz")
        max_diff = float(np.max(np.abs(ref["pred_mixed"] - arrays[0][:, :, 0])))
        pred64, true64 = arrays[0][:, :, 0], arrays[3][:, :, 0]
        mse64, mae64 = metrics_f64(pred64, true64)
        m32 = json.load(open(gdir / "test_metrics_after_train.json"))
        recompute = {
            "mse_f64": mse64, "mae_f64": mae64,
            "mse_f32_main": m32["mse"], "mae_f32_main": m32["mae"],
            "abs_delta_mse": abs(mse64 - m32["mse"]),
            "abs_delta_mae": abs(mae64 - m32["mae"]),
            "reload_max_abs_pred_diff": max_diff,
            "reload_tol": args.reload_tol,
            "reload_pass": bool(max_diff <= args.reload_tol),
        }
        (gdir / "recompute_metrics.json").write_text(
            json.dumps(recompute, indent=2))
        (gdir / "reload_check.json").write_text(json.dumps(
            {"max_abs_diff": max_diff, "tol": args.reload_tol,
             "pass": bool(max_diff <= args.reload_tol)}, indent=2))
        print("RELOAD max|d|=%.3e pass=%s ; f64 mse=%.8f (d=%.2e)"
              % (max_diff, recompute["reload_pass"], mse64,
                 recompute["abs_delta_mse"]), flush=True)
        if not recompute["reload_pass"]:
            sys.exit(2)
        if recompute["abs_delta_mse"] > 1e-6:
            print("[WARN] float64 recompute delta > 1e-6", flush=True)


if __name__ == "__main__":
    main()
