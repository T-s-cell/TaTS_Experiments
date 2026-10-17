"""Summarize the 36-group TaTS campaign + gate verification sweep.

Usage:
  python -m repro_tats.summarize --out_root <dir> --protocol_dir <dir>

Per group verifies (exit 2 on any failure):
  - status.json says done; all artifacts present
  - reload_check.json pass with max|d| <= 1e-6 (REAL baseline comparison)
  - recompute float64 |dMSE| vs train baseline <= 1e-6
  - run_config.json explicit caps (max_batches / max_eval_batches)
  - test_arrays.npz full coverage: window_index == 0..n-1 exactly once and
    count == splits.json expected
  - protocol files hash-match the group manifest

Outputs:
  <out_root>/summary.json      machine-readable (per-group + per-domain)
  <out_root>/summary.md        human-readable tables

Metrics are in the train-standardized space (per protocol): full-element
float32 MSE/MAE over all test windows; per-domain = equal-weight mean of the
domain's 4 horizons.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

_ARTIFACTS = ["checkpoint.pth", "manifest.json", "history.json",
              "test_arrays.npz", "test_arrays_reload.npz",
              "test_metrics_after_train.json", "test_metrics_reload.json",
              "recompute_metrics.json", "reload_check.json",
              "train_phase.json", "run_config.json"]
DOMAINS = ["Agriculture", "Climate", "Economy", "Energy", "Environment",
           "Health", "Security", "SocialGood", "Traffic"]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--protocol_dir", required=True)
    ap.add_argument("--seed", type=int, default=2021)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    status = json.load(open(out_root / "status.json"))["groups"]
    splits = json.load(open(Path(args.protocol_dir) / "splits.json"))

    groups, failures = [], []
    for dom in DOMAINS:
        for h in splits[dom]["H"]:
            gid = "%s_H%d_seed%d" % (dom, h, args.seed)
            gdir = out_root / "runs" / gid
            row = {"group": gid, "domain": dom, "horizon": h}
            try:
                for f in _ARTIFACTS:
                    if not (gdir / f).exists():
                        raise ValueError("missing artifact %s" % f)
                if status.get(gid, {}).get("state") != "done":
                    raise ValueError("status %s != done"
                                     % status.get(gid, {}).get("state"))
                rc = json.load(open(gdir / "reload_check.json"))
                if not rc["pass"] or rc["max_abs_diff"] > 1e-6:
                    raise ValueError("reload max|d|=%s pass=%s"
                                     % (rc["max_abs_diff"], rc["pass"]))
                rec = json.load(open(gdir / "recompute_metrics.json"))
                if rec["abs_delta_mse_vs_baseline"] > 1e-6:
                    raise ValueError("f64 delta %s"
                                     % rec["abs_delta_mse_vs_baseline"])
                cfgc = json.load(open(gdir / "run_config.json"))
                if cfgc["max_batches"] or cfgc["max_eval_batches"]:
                    raise ValueError("caps not zero: %d/%d"
                                     % (cfgc["max_batches"],
                                        cfgc["max_eval_batches"]))
                arr = np.load(gdir / "test_arrays.npz")
                idx = arr["window_index"]
                want = int(splits[dom]["window_counts"][str(h)]["test"])
                if len(idx) != want or not np.array_equal(
                        np.sort(idx), np.arange(want)):
                    raise ValueError("coverage %d != %d" % (len(idx), want))
                man = json.load(open(gdir / "manifest.json"))
                for rel, sha in man["protocol_sha256"].items():
                    if file_sha256(Path(args.protocol_dir) / rel) != sha:
                        raise ValueError("protocol drift %s" % rel)
                m = json.load(open(gdir / "test_metrics_after_train.json"))
                hist = json.load(open(gdir / "history.json"))
                tp = json.load(open(gdir / "train_phase.json"))
                row.update({
                    "mse": m["mse"], "mae": m["mae"],
                    "windows": m["evaluated_windows"],
                    "best_epoch": tp["best_epoch"],
                    "epochs_run": tp["epochs_run"],
                    "early_stop": bool(hist[-1].get("early_stop", False))
                    if "early_stop" in hist[-1] else None,
                    "train_seconds": tp["train_wall_seconds"],
                    "peak_mem_gib": tp["peak_mem_gib"],
                    "reload_max_abs_diff": rc["max_abs_diff"],
                    "f64_mse": rec["mse_f64_this_phase"],
                    "f64_delta_mse": rec["abs_delta_mse_vs_baseline"],
                })
            except Exception as e:  # noqa: BLE001 - collect and continue
                row["error"] = str(e)
                failures.append((gid, str(e)))
            groups.append(row)

    by_domain = {}
    for dom in DOMAINS:
        rows = [g for g in groups if g["domain"] == dom and "mse" in g]
        if len(rows) == 4:
            by_domain[dom] = {
                "mse": float(np.mean([g["mse"] for g in rows])),
                "mae": float(np.mean([g["mae"] for g in rows])),
                "train_seconds": float(np.sum([g["train_seconds"]
                                               for g in rows])),
            }
    ok = [g for g in groups if "mse" in g]
    overall = {
        "mse": float(np.mean([g["mse"] for g in ok])),
        "mae": float(np.mean([g["mae"] for g in ok])),
        "groups_ok": len(ok), "groups_total": len(groups),
        "test_windows": int(sum(g["windows"] for g in ok)),
        "train_seconds_total": float(sum(g["train_seconds"] for g in ok)),
    }

    result = {"groups": groups, "by_domain": by_domain, "overall": overall,
              "failures": failures}
    (out_root / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))

    lines = ["# TaTS x timemmd_common_801010_v1 — 36 组汇总", "",
             "| 组 | 测试窗 | MSE | MAE | best_ep | epochs | train_s | reload_max|d| | f64 dMSE |",
             "|---|---|---|---|---|---|---|---|---|"]
    for g in groups:
        if "mse" in g:
            lines.append(
                "| %s | %d | %.6f | %.6f | %s | %s | %.1f | %.1e | %.1e |"
                % (g["group"], g["windows"], g["mse"], g["mae"],
                   g["best_epoch"], g["epochs_run"], g["train_seconds"],
                   g["reload_max_abs_diff"], g["f64_delta_mse"]))
        else:
            lines.append("| %s | ERROR: %s |" % (g["group"], g["error"]))
    lines += ["", "## 逐域 4H 等权均值", "",
              "| 域 | MSE | MAE |", "|---|---|---|"]
    for dom, d in by_domain.items():
        lines.append("| %s | %.6f | %.6f |" % (dom, d["mse"], d["mae"]))
    lines += ["", "## Overall(36 组等权)", "",
              "- MSE %.6f | MAE %.6f | 测试窗 %d | 训练总耗时 %.1f s"
              % (overall["mse"], overall["mae"], overall["test_windows"],
                 overall["train_seconds_total"]),
              "", "Failures: %d" % len(failures), ""]
    (out_root / "summary.md").write_text("\n".join(lines))

    print("\n".join(lines[-12:]))
    if failures:
        for gid, e in failures:
            print("FAIL %s: %s" % (gid, e))
        sys.exit(2)
    print("SUMMARY OK: %d groups, all gates pass" % len(groups))


if __name__ == "__main__":
    main()
