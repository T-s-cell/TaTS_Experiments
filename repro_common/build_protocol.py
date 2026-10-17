"""Build the unified TimeMMD protocol `timemmd_common_801010_v1`.

Pure-CPU, model-free. Produces dataset/timemmd_common_801010_v1/:
  <Domain>.csv x9          (all original columns kept; + orig_row; Health + date_fixed)
  scaler.json              (per-domain OT StandardScaler fit on train segment only)
  splits.json              (per-domain N/L/H/freq/n_train/val_end/n_test + window counts)
  windows/<Domain>_H<h>.csv x36  (explicit window manifests, split column included)
  health_date_fix_map.csv  (YEAR, WEEK, old date, new date, orig_row)
  economy_frozen_evidence/ (EXP-020 frozen sorted CSV + sort_mapping + original)
  protocol.json            (field maps, provenance, sha256 of every file above)

Conventions fixed here and independently asserted by window_check.py:
  - orig_row is the 0-based data row number in the ORIGINAL unsorted raw file
    (header excluded). For Economy, rows are written in EXP-020 frozen sorted
    order and orig_row comes from the frozen sort_mapping (verified full-row
    integrity against the raw file).
  - Health keeps all 1389 rows; `date_fixed` = (Sunday of MMWR week 1, which
    contains Jan 4) + 7*(WEEK-1) + 1 day. The original `date` column is kept
    untouched; loaders read date_fixed.
  - fact empty string -> "No information available"; prior_history_avg is kept
    exactly as provided (verified: no empty values in any domain).
  - Original column names are never rewritten (Agriculture/Traffic keep both
    `Date` and `date`; asserted identical row-by-row).
"""
import hashlib
import json
import os
import shutil
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.environ.get("TIMEMMD_RAW", "/home/wlt/MMTS/data/TimeMMD")
OUT_DIR = os.path.join(REPO, "dataset", "timemmd_common_801010_v1")
FROZEN_DIR = os.path.join(REPO, "dataset", "economy_frozen_evidence")

FROZEN_EXPECT = {
    "Economy_frozen_sorted.csv": "023292b969b915a73185e047debfb18f5572544f9463b23e64e9f496a0b52b26",
    "sort_mapping.csv": "43c3fd29f6275eb346066c2d3d6e2182f4f174bfd9d8bd6c6ab75d0492cc74b2",
    "Economy_original.csv": "c35325f0181a426a78c35b9172d8908f64639796a535c5fdc8b082b538030e5d",
}
ECONOMY_INVERSIONS_RAW = 35
HEALTH_CHANGED = 832
HEALTH_RANGE = ("1997-09-29", "2024-05-06")
HEALTH_CHECKPOINTS = {(1998, 1): "1998-01-05", (2021, 1): "2021-01-04"}
NO_INFO = "No information available"

DOMS = {
    "Agriculture": dict(L=192, H=[6, 8, 10, 12], freq="ME"),
    "Climate": dict(L=192, H=[6, 8, 10, 12], freq="ME"),
    "Economy": dict(L=192, H=[6, 8, 10, 12], freq="ME"),
    "Energy": dict(L=1056, H=[12, 24, 36, 48], freq="W"),
    "Environment": dict(L=528, H=[48, 96, 192, 336], freq="D"),
    "Health": dict(L=96, H=[12, 24, 36, 48], freq="W"),
    "Security": dict(L=220, H=[6, 8, 10, 12], freq="ME"),
    "SocialGood": dict(L=192, H=[6, 8, 10, 12], freq="ME"),
    "Traffic": dict(L=96, H=[6, 8, 10, 12], freq="ME"),
}


def fail(msg):
    print("[FATAL] " + msg)
    sys.exit(2)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def timef_dim(freq):
    sys.path.insert(0, REPO)
    from utils.timefeatures import time_features_from_frequency_str
    return len(time_features_from_frequency_str(freq))


def mmwr_monday(year, week):
    jan4 = date(year, 1, 4)
    week1_sunday = jan4 - timedelta(days=(jan4.weekday() + 1) % 7)
    return week1_sunday + timedelta(days=7 * (week - 1) + 1)


def window_counts(n, L, H):
    n_train = 4 * n // 5
    n_test = n // 10
    val_end = n - n_test
    return n_train, val_end, n_test, {
        "train": (n_train - H) - L + 1,
        "val": (val_end - H) - n_train + 1,
        "test": (n - H) - val_end + 1,
    }


def main():
    sys.path.insert(0, REPO)
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"protocol": "timemmd_common_801010_v1",
              "orig_row_convention": "0-based data row in the original unsorted "
                                     "raw CSV (header excluded)",
              "domains": {}}
    scalers = {}
    inversions = None

    # ---- Economy frozen evidence: verify hashes, copy into protocol dir ----
    ev_dir = os.path.join(OUT_DIR, "economy_frozen_evidence")
    os.makedirs(ev_dir, exist_ok=True)
    for name, want in FROZEN_EXPECT.items():
        src = os.path.join(FROZEN_DIR, name)
        got = sha256(src)
        if got != want:
            fail("Economy frozen evidence %s sha256 %s != expected %s" % (name, got, want))
        shutil.copyfile(src, os.path.join(ev_dir, name))
    frozen = pd.read_csv(os.path.join(ev_dir, "Economy_frozen_sorted.csv"),
                         dtype=str, keep_default_na=False)
    mapping = pd.read_csv(os.path.join(ev_dir, "sort_mapping.csv"))
    raw_econ = pd.read_csv(os.path.join(RAW_DIR, "Economy.csv"),
                           dtype=str, keep_default_na=False)
    if not (len(frozen) == len(mapping) == len(raw_econ) == 423):
        fail("Economy row count mismatch")
    if list(mapping["new_row_index"]) != list(range(423)):
        fail("sort_mapping new_row_index is not 0..422 in order")
    econ_orig = [int(r) for r in mapping["original_row_index"]]
    if sorted(econ_orig) != list(range(423)):
        fail("sort_mapping original_row_index is not a permutation")
    for i, orow in enumerate(econ_orig):
        if not frozen.iloc[i].equals(raw_econ.iloc[orow]):
            fail("Economy frozen row %d != raw row %d (EXP-020 mapping broken)" % (i, orow))
    edates = pd.to_datetime(raw_econ["date"], format="%Y-%m-%d")
    inversions = int((edates.diff().dt.days < 0).sum())
    if inversions != ECONOMY_INVERSIONS_RAW:
        fail("Economy raw adjacent inversions %d != %d" % (inversions, ECONOMY_INVERSIONS_RAW))
    fdates = pd.to_datetime(frozen["date"], format="%Y-%m-%d")
    if not fdates.is_monotonic_increasing:
        fail("Economy frozen order not ascending")

    for dom, spec in DOMS.items():
        print("== %s ==" % dom)
        raw_path = os.path.join(RAW_DIR, dom + ".csv")
        raw_sha = sha256(raw_path)
        df = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
        n = len(df)
        for col in ("OT", "prior_history_avg", "fact", "date"):
            if col not in df.columns:
                fail("%s missing column %s" % (dom, col))

        if "Date" in df.columns:
            fmt = "%Y%m" if df["Date"].str.fullmatch(r"\d{6}").all() else "%Y-%m-%d"
            d_up = pd.to_datetime(df["Date"], format=fmt, errors="raise")
            if not (d_up.values == pd.to_datetime(
                    df["date"], format="%Y-%m-%d").values).all():
                fail("%s Date(%s) != date" % (dom, fmt))

        fact_empty = int((df["fact"].str.strip() == "").sum())
        df.loc[df["fact"].str.strip() == "", "fact"] = NO_INFO
        prior_empty = int((df["prior_history_avg"].str.strip() == "").sum())
        if prior_empty:
            fail("%s has %d empty prior_history_avg" % (dom, prior_empty))

        if dom == "Economy":
            df = frozen.copy()
            df["orig_row"] = econ_orig
            dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
            if not dates.is_monotonic_increasing or dates.nunique() != n:
                fail("Economy final dates not ascending-unique")
        elif dom == "Health":
            year = df["YEAR"].astype(int)
            week = df["WEEK"].astype(int)
            df["date_fixed"] = [mmwr_monday(int(y), int(w)).isoformat()
                                for y, w in zip(year, week)]
            new_ts = pd.to_datetime(df["date_fixed"])
            old_ts = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
            changed = int((new_ts.values != old_ts.values).sum())
            diffs = sorted(set(new_ts.diff().dt.days.dropna().astype(int)))
            print("  Health: changed=%d range=%s..%s uniq_new=%d uniq_old=%d "
                  "adjacent_days=%s" % (changed, df["date_fixed"].iloc[0],
                                        df["date_fixed"].iloc[-1],
                                        new_ts.nunique(), old_ts.nunique(), diffs))
            if changed != HEALTH_CHANGED:
                fail("Health %d rows changed != %d" % (changed, HEALTH_CHANGED))
            if (df["date_fixed"].iloc[0], df["date_fixed"].iloc[-1]) != HEALTH_RANGE:
                fail("Health fixed range mismatch")
            if new_ts.nunique() != n:
                fail("Health fixed dates not unique")
            if diffs != [7]:
                fail("Health adjacent diffs not all 7 days: %s" % diffs)
            if old_ts.nunique() != n - 5:
                fail("Health old unique dates != n-5 (5 dup groups expected)")
            for (y, w), want in HEALTH_CHECKPOINTS.items():
                got = mmwr_monday(y, w).isoformat()
                if got != want:
                    fail("Health checkpoint %d-W%02d %s != %s" % (y, w, got, want))
            pd.DataFrame({"YEAR": year, "WEEK": week, "old_date": df["date"],
                          "new_date": df["date_fixed"], "orig_row": econ_orig
                          if dom == "Economy" else list(range(n))}).to_csv(
                os.path.join(OUT_DIR, "health_date_fix_map.csv"), index=False)
            dates = new_ts
        else:
            dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
            if not dates.is_monotonic_increasing:
                fail("%s raw dates not ascending" % dom)
            if dates.nunique() != n:
                fail("%s dates not unique" % dom)

        df["orig_row"] = list(range(n)) if dom != "Economy" else econ_orig
        n_train = 4 * n // 5
        n_test = n // 10
        val_end = n - n_test

        ot = np.array([float(v) for v in df["OT"]], dtype=np.float64)
        if not np.isfinite(ot).all():
            fail("%s OT not finite" % dom)
        scaler = {"mean": float(ot[:n_train].mean()),
                  "std": float(ot[:n_train].std(ddof=0))}
        if scaler["std"] <= 0:
            fail("%s degenerate train OT std" % dom)
        scalers[dom] = scaler

        df.to_csv(os.path.join(OUT_DIR, dom + ".csv"), index=False)

        counts = {}
        wdir = os.path.join(OUT_DIR, "windows")
        os.makedirs(wdir, exist_ok=True)
        for H in spec["H"]:
            counts[str(H)] = window_counts(n, spec["L"], H)[3]
            rows = []
            for split, lo, hi in (("train", spec["L"], n_train - H),
                                  ("val", n_train, val_end - H),
                                  ("test", val_end, n - H)):
                for s in range(lo, hi + 1):
                    rows.append({
                        "domain": dom, "H": H, "split": split, "s": s,
                        "orig_row": int(df["orig_row"].iloc[s]),
                        "input_date_range": json.dumps(
                            [str(dates.iloc[s - spec["L"]].date()),
                             str(dates.iloc[s - 1].date())]),
                        "label_dates": json.dumps(
                            [str(dates.iloc[s + k].date()) for k in range(H)]),
                    })
            pd.DataFrame(rows).to_csv(
                os.path.join(wdir, "%s_H%d.csv" % (dom, H)), index=False)

        report["domains"][dom] = {
            "columns": list(pd.read_csv(raw_path, dtype=str,
                                        keep_default_na=False).columns),
            "field_map": {"date": "date_fixed" if dom == "Health" else "date",
                          "target": "OT", "fact": "fact",
                          "prior": "prior_history_avg"},
            "N": n, "L": spec["L"], "H": spec["H"], "freq": spec["freq"],
            "timef_dim": timef_dim(spec["freq"]),
            "n_train": n_train, "val_end": val_end, "n_test": n_test,
            "window_counts": counts,
            "fact_filled": fact_empty,
            "prior_history_avg_empty": prior_empty,
            "raw_csv_sha256": raw_sha,
        }
        print("  N=%d L=%d H=%s freq=%s timef_dim=%d fact_filled=%d counts=%s"
              % (n, spec["L"], spec["H"], spec["freq"], timef_dim(spec["freq"]),
                 fact_empty, json.dumps(counts)))

    with open(os.path.join(OUT_DIR, "splits.json"), "w") as f:
        json.dump({d: {k: v for k, v in r.items() if k != "columns"}
                   for d, r in report["domains"].items()}, f, indent=2)
    with open(os.path.join(OUT_DIR, "scaler.json"), "w") as f:
        json.dump(scalers, f, indent=2)

    files_sha = {}
    for root, _, names in os.walk(OUT_DIR):
        for name in sorted(names):
            p = os.path.join(root, name)
            rel = os.path.relpath(p, OUT_DIR)
            if rel == "protocol.json":
                continue
            files_sha[rel] = sha256(p)
    report["files_sha256"] = files_sha
    report["raw_dir"] = RAW_DIR
    report["economy_frozen_expected_sha256"] = FROZEN_EXPECT
    report["economy_inversions_raw"] = inversions
    with open(os.path.join(OUT_DIR, "protocol.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("protocol written to", OUT_DIR)


if __name__ == "__main__":
    main()
