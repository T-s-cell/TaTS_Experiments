"""Run one TaTS reproduction group end-to-end.

Modes:
  train       train -> best-checkpoint test -> save BASELINE arrays/metrics
              (test_arrays.npz is written once here and never overwritten by
              any later phase), then spawn a NEW process (--mode test_only)
              that reloads the checkpoint and verifies predictions AND window
              identity against the baseline within tolerance, plus float64
              recompute.
  test_only   independent reload verification (no training): checkpoint ->
              predict -> compare vs the TRAIN-PHASE BASELINE npz -> writes
              test_arrays_reload.npz / reload_check.json / recompute_metrics.json
              (float64). Never writes test_arrays.npz or run_config.json.
  --perturb P (test_only): add P to reloaded pred_mixed BEFORE comparison and
              write *_perturb artifacts instead. Negative control: the
              comparator MUST fail. Exit 0 = correctly detected, exit 3 =
              comparator failed to detect (comparator broken).

Gates (exit 2): any reload diff > tol, window-identity mismatch, non-natural
stop, missing artifacts, or train run_config mismatch in test_only.
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
BASELINE_NPZ = "test_arrays.npz"
COMPARE_KEYS = ["pred_mixed", "pred_model", "prior_y", "truth",
                "s", "orig_row", "label_dates", "window_index"]
IDENTITY_KEYS = ["s", "orig_row", "label_dates", "window_index"]


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


def save_test_arrays(path, arrays, identity):
    np.savez_compressed(
        path,
        pred_mixed=arrays[0][:, :, 0], pred_model=arrays[1][:, :, 0],
        prior_y=arrays[2][:, :, 0], truth=arrays[3][:, :, 0],
        s=identity["s"], orig_row=identity["orig_row"],
        label_dates=identity["label_dates"], window_index=arrays[4])


def compare_reload(gdir, arrays, identity, tol):
    """Compare reload-phase predictions + window identity against the
    train-phase BASELINE npz. Returns a verdict dict; NEVER writes the
    baseline."""
    ref = np.load(Path(gdir) / BASELINE_NPZ)
    rel = {
        "pred_mixed": arrays[0][:, :, 0], "pred_model": arrays[1][:, :, 0],
        "prior_y": arrays[2][:, :, 0], "truth": arrays[3][:, :, 0],
        "s": identity["s"], "orig_row": identity["orig_row"],
        "label_dates": identity["label_dates"], "window_index": arrays[4],
    }
    per, fatal = {}, None
    for k in COMPARE_KEYS:
        r, v = ref[k], rel[k]
        if r.shape != v.shape:
            fatal = "shape mismatch %s: baseline %s vs reload %s" % (
                k, r.shape, v.shape)
            break
        if k in IDENTITY_KEYS or r.dtype.kind in "US":
            per[k] = bool(np.array_equal(r, v))
        else:
            d = np.abs(r.astype(np.float64) - v.astype(np.float64))
            per[k] = float(d.max()) if d.size else 0.0
    if fatal is None:
        max_diff = max(v for k, v in per.items() if isinstance(v, float))
        identity_ok = all(per[k] is True for k in IDENTITY_KEYS)
        return {"per_key_max_abs_diff_or_equal": per,
                "max_abs_diff": max_diff, "tol": tol,
                "identity_pass": bool(identity_ok),
                "pass": bool(identity_ok and max_diff <= tol)}
    return {"error": fatal, "pass": False}


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


def do_test_phase(exp, gdir, tag, arrays_name):
    data, loader = exp._get_data("test")
    arrays = exp.predict(data, loader)
    idx = arrays[4].tolist()
    identity = data.get_window_identity(idx)
    save_test_arrays(Path(gdir) / arrays_name, arrays, identity)
    mse, mae, per_step = metrics_f32(arrays[0][:, :, 0], arrays[3][:, :, 0])
    if not all(np.isfinite(v) for v in (mse, mae)):
        raise ValueError("non-finite test metrics")
    result = {"tag": tag, "mse": mse, "mae": mae,
              "evaluated_windows": len(arrays[0]),
              "expected_windows": len(data),
              "per_step": per_step}
    if not exp.cfg.max_eval_batches and len(arrays[0]) != len(data):
        raise ValueError("test coverage %d != %d"
                         % (len(arrays[0]), len(data)))
    (Path(gdir) / ("test_metrics_%s.json" % tag)).write_text(
        json.dumps(result, indent=2))
    return arrays, identity


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
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="test_only negative control: add this to reloaded "
                         "pred_mixed before comparison; comparator must FAIL")
    args = ap.parse_args()

    cfg = build_cfg(args)
    gid = cfg.group_id()
    gdir = Path(cfg.out_root) / "runs" / gid
    gdir.mkdir(parents=True, exist_ok=True)
    ckpt = gdir / "checkpoint.pth"
    run_config_path = gdir / "run_config.json"

    seed_everything(cfg.seed)
    exp = Exp_TaTS(cfg)

    if args.mode == "train":
        if ckpt.exists():
            raise SystemExit("[FATAL] refusing to overwrite %s" % ckpt)
        run_config_path.write_text(
            json.dumps(cfg.public_dict(), indent=2))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        best = exp.train(str(gdir))
        train_wall = time.time() - t0
        do_test_phase(exp, gdir, "after_train", BASELINE_NPZ)
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
             "--max_batches", str(args.max_batches),
             "--max_eval_batches", str(args.max_eval_batches),
             "--reload_tol", str(args.reload_tol)],
            env=env, cwd=str(Path(__file__).resolve().parents[1]))
        if r.returncode != 0:
            raise SystemExit("[FATAL] reload/recompute phase failed rc=%d"
                             % r.returncode)
        print("GROUP %s DONE" % gid, flush=True)

    else:  # test_only: fresh process, baseline-vs-reload verification
        if run_config_path.exists():
            saved = json.load(open(run_config_path))
            now = cfg.public_dict()
            if saved != now:
                diff = {k: (saved.get(k, "<missing>"), now.get(k, "<missing>"))
                        for k in set(saved) | set(now)
                        if saved.get(k) != now.get(k)}
                raise SystemExit("[FATAL] run_config.json mismatch: %s" % diff)
        else:
            raise SystemExit("[FATAL] run_config.json missing (train phase "
                             "must run first): %s" % run_config_path)
        exp.load_checkpoint(str(ckpt))
        perturbed = args.perturb != 0.0
        arrays, identity = do_test_phase(
            exp, gdir, "reload_perturb" if perturbed else "reload",
            "test_arrays_reload_perturb.npz" if perturbed
            else "test_arrays_reload.npz")
        if perturbed:
            arrays = (arrays[0] + args.perturb, arrays[1], arrays[2],
                      arrays[3], arrays[4])
        verdict = compare_reload(gdir, arrays, identity, args.reload_tol)
        pred64, true64 = arrays[0][:, :, 0], arrays[3][:, :, 0]
        mse64, mae64 = metrics_f64(pred64, true64)
        m32 = json.load(open(gdir / "test_metrics_after_train.json"))
        recompute = {
            "perturb": args.perturb,
            "mse_f64_this_phase": mse64, "mae_f64_this_phase": mae64,
            "mse_f32_train_baseline": m32["mse"],
            "mae_f32_train_baseline": m32["mae"],
            "abs_delta_mse_vs_baseline": abs(mse64 - m32["mse"]),
            "abs_delta_mae_vs_baseline": abs(mae64 - m32["mae"]),
            **verdict,
        }
        suffix = "_perturb" if perturbed else ""
        (gdir / ("recompute_metrics%s.json" % suffix)).write_text(
            json.dumps(recompute, indent=2))
        (gdir / ("reload_check%s.json" % suffix)).write_text(
            json.dumps({"max_abs_diff": verdict.get("max_abs_diff"),
                        "tol": args.reload_tol,
                        "pass": verdict["pass"]}, indent=2))
        print("RELOAD%s max|d|=%s pass=%s ; f64 mse=%.8f (d=%.2e)"
              % (suffix, verdict.get("max_abs_diff", "n/a"),
                 verdict["pass"], mse64,
                 recompute["abs_delta_mse_vs_baseline"]), flush=True)
        if perturbed:
            # negative control: comparator MUST detect the perturbation
            if verdict["pass"]:
                print("[COMPARATOR BROKEN] perturbation not detected",
                      flush=True)
                sys.exit(3)
            print("NEGATIVE CONTROL OK (perturbation detected)", flush=True)
        else:
            if not verdict["pass"]:
                sys.exit(2)
            if recompute["abs_delta_mse_vs_baseline"] > 1e-6:
                print("[WARN] float64 recompute delta > 1e-6", flush=True)


if __name__ == "__main__":
    main()
