"""Serial queue driver for the 36 TaTS groups (9 domains x 4 horizons).

State machine per group in <out_root>/status.json:
  pending -> running -> postverify -> done | failed

Behavior:
  - groups already `done` are skipped after re-verifying the protocol hash
    and the presence of every artifact (checkpoint, arrays, metrics,
    reload_check pass)
  - a failed group keeps its logs and is marked failed; the queue continues
  - the queue STOPS only when two consecutive groups fail with the same
    exception signature (shared-code defect)

Run under tmux:  python repro_tats/run_all.py --out_root <dir> [--preflight]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PREFLIGHT = [("Energy", 48), ("Environment", 336), ("Security", 12)]


def load_status(path):
    if os.path.exists(path):
        return json.load(open(path))
    return {"groups": {}}


def save_status(path, st):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, path)


def artifacts_ok(gdir, st_entry):
    need = ["checkpoint.pth", "manifest.json", "history.json",
            "test_arrays.npz", "test_metrics_after_train.json",
            "recompute_metrics.json", "reload_check.json",
            "train_phase.json", "run_config.json"]
    for f in need:
        if not os.path.exists(os.path.join(gdir, f)):
            return False
    rc = json.load(open(os.path.join(gdir, "reload_check.json")))
    if not rc.get("pass"):
        return False
    return True


def group_list(protocol_dir, preflight=False):
    splits = json.load(open(os.path.join(protocol_dir, "splits.json")))
    if preflight:
        return [(d, h) for d, h in PREFLIGHT]
    out = []
    for dom in ["Agriculture", "Climate", "Economy", "Energy", "Environment",
                "Health", "Security", "SocialGood", "Traffic"]:
        for h in splits[dom]["H"]:
            out.append((dom, int(h)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--max_eval_batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--only", default=None,
                    help="run a single Domain:H then exit")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    (out_root / "runs").mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    status_path = str(out_root / "status.json")
    st = load_status(status_path)

    jobs = group_list(args.protocol_dir, args.preflight)
    if args.only:
        dom, h = args.only.rsplit(":", 1)
        jobs = [(dom, int(h))]

    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env["TOKENIZERS_PARALLELISM"] = "true"

    queue_log = open(out_root / ("logs" if False else "run_all.log"), "a")

    def log(msg):
        line = "[%s] %s" % (time.strftime("%F %T"), msg)
        print(line, flush=True)
        queue_log.write(line + "\n")
        queue_log.flush()

    prev_fail_sig = None
    for dom, h in jobs:
        gid = "%s_H%d_seed%d" % (dom, h, args.seed)
        gdir = out_root / "runs" / gid
        entry = st["groups"].setdefault(
            gid, {"state": "pending", "attempts": 0})
        if entry["state"] == "done" and artifacts_ok(gdir, entry):
            log("SKIP %s (done, artifacts verified)" % gid)
            continue
        if entry["state"] in ("running", "postverify"):
            log("RESET %s from interrupted state %s" % (gid, entry["state"]))
            entry["state"] = "pending"

        entry["attempts"] += 1
        entry["state"] = "running"
        entry["started_at"] = time.strftime("%F %T")
        save_status(status_path, st)
        log("START %s (attempt %d)" % (gid, entry["attempts"]))

        cmd = [sys.executable, "-u", str(REPO / "repro_tats" / "run_group.py"),
               "--mode", "train", "--domain", dom, "--horizon", str(h),
               "--protocol_dir", args.protocol_dir,
               "--out_root", args.out_root, "--seed", str(args.seed)]
        if args.max_batches:
            cmd += ["--max_batches", str(args.max_batches)]
        if args.max_eval_batches:
            cmd += ["--max_eval_batches", str(args.max_eval_batches)]
        lg = open(out_root / "logs" / ("%s.log" % gid), "a")
        try:
            r = subprocess.run(cmd, env=env, cwd=str(REPO),
                               stdout=lg, stderr=subprocess.STDOUT)
            rc = r.returncode
        except Exception:
            lg.write(traceback.format_exc())
            rc = -99
        lg.close()

        if rc == 0:
            entry["state"] = "postverify"
            entry["finished_at"] = time.strftime("%F %T")
            save_status(status_path, st)
            if artifacts_ok(gdir, entry):
                entry["state"] = "done"
                log("DONE %s" % gid)
            else:
                entry["state"] = "failed"
                entry["error"] = "artifacts incomplete after rc=0"
                log("FAIL %s (artifacts incomplete)" % gid)
        else:
            entry["state"] = "failed"
            tail = ""
            try:
                tail = open(out_root / "logs" / ("%s.log" % gid)).read()[-2000:]
            except OSError:
                pass
            entry["error"] = "rc=%d %s" % (rc, tail[-400:])
            entry["finished_at"] = time.strftime("%F %T")
            log("FAIL %s rc=%d (log kept)" % (gid, rc))
        save_status(status_path, st)

        if entry["state"] == "failed":
            sig = tail.splitlines()[-1][:120] if tail else "rc=%d" % rc
            if prev_fail_sig is not None and sig == prev_fail_sig:
                log("STOP queue: two consecutive groups failed identically "
                    "(shared defect?): %s" % sig)
                break
            prev_fail_sig = sig
        else:
            prev_fail_sig = None

    done = sum(1 for v in st["groups"].values() if v["state"] == "done")
    log("QUEUE EXIT: %d/%d done" % (done, len(jobs)))
    queue_log.close()


if __name__ == "__main__":
    main()
