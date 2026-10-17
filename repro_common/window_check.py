"""Independent verification of protocol timemmd_common_801010_v1.

Deliberately shares no code or constants with build_protocol.py: the expected
window-count table below is transcribed from the directive's acceptance values
(user-corrected) and every artifact is recomputed from the written files.
Exit 0 = all checks pass, exit 2 = any failure (stops the campaign).
"""
import hashlib
import json
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.environ.get(
    "PROTOCOL_DIR",
    os.path.join(REPO, "dataset", "timemmd_common_801010_v1"))
RAW_DIR = os.environ.get("TIMEMMD_RAW", "/home/wlt/MMTS/data/TimeMMD")
NO_INFO = "No information available"

# (train, val, test) per (domain, H) -- from the directive's acceptance table
EXPECTED = {
    "Agriculture": {6: (199, 46, 44), 8: (197, 44, 42), 10: (195, 42, 40), 12: (193, 40, 38)},
    "Climate":     {6: (199, 46, 44), 8: (197, 44, 42), 10: (195, 42, 40), 12: (193, 40, 38)},
    "Economy":     {6: (141, 38, 37), 8: (139, 36, 35), 10: (137, 34, 33), 12: (135, 32, 31)},
    "Energy":      {12: (116, 138, 136), 24: (104, 126, 124), 36: (92, 114, 112), 48: (80, 102, 100)},
    "Environment": {48: (11623, 1479, 1477), 96: (11575, 1431, 1429),
                    192: (11479, 1335, 1333), 336: (11335, 1191, 1189)},
    "Health":      {12: (1004, 129, 127), 24: (992, 117, 115), 36: (980, 105, 103), 48: (968, 93, 91)},
    "Security":    {6: (12, 26, 24), 8: (10, 24, 22), 10: (8, 22, 20), 12: (6, 20, 18)},
    "SocialGood":  {6: (516, 85, 84), 8: (514, 83, 82), 10: (512, 81, 80), 12: (510, 79, 78)},
    "Traffic":     {6: (323, 49, 48), 8: (321, 47, 46), 10: (319, 45, 44), 12: (317, 43, 42)},
}
FROZEN_SHA = {
    "economy_frozen_evidence/Economy_frozen_sorted.csv":
        "023292b969b915a73185e047debfb18f5572544f9463b23e64e9f496a0b52b26",
    "economy_frozen_evidence/sort_mapping.csv":
        "43c3fd29f6275eb346066c2d3d6e2182f4f174bfd9d8bd6c6ab75d0492cc74b2",
    "economy_frozen_evidence/Economy_original.csv":
        "c35325f0181a426a78c35b9172d8908f64639796a535c5fdc8b082b538030e5d",
}

fails = []
checks = 0


def check(cond, msg):
    global checks
    checks += 1
    if not cond:
        fails.append(msg)
        print("[FAIL] " + msg)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mmwr_monday(year, week):
    jan4 = date(year, 1, 4)
    return jan4 - timedelta(days=(jan4.weekday() + 1) % 7) \
        + timedelta(days=7 * (week - 1) + 1)


def main():
    splits = json.load(open(os.path.join(OUT_DIR, "splits.json")))
    scaler = json.load(open(os.path.join(OUT_DIR, "scaler.json")))
    proto = json.load(open(os.path.join(OUT_DIR, "protocol.json")))

    total_test = 0
    for dom, exp_h in EXPECTED.items():
        s = splits[dom]
        n, L = s["N"], s["L"]
        n_train, n_test, val_end = 4 * n // 5, n // 10, n - n // 10
        check(s["n_train"] == n_train and s["n_test"] == n_test
              and s["val_end"] == val_end, "%s split borders" % dom)
        check(sorted(s["H"]) == sorted(exp_h.keys()), "%s H set" % dom)

        df = pd.read_csv(os.path.join(OUT_DIR, dom + ".csv"),
                         dtype=str, keep_default_na=False)
        check(len(df) == n, "%s final N=%d != %d" % (dom, len(df), n))
        if "Date" in df.columns:
            fmt = "%Y%m" if df["Date"].str.fullmatch(r"\d{6}").all() else "%Y-%m-%d"
            check((pd.to_datetime(df["Date"], format=fmt).values
                   == pd.to_datetime(df["date"], format="%Y-%m-%d").values).all(),
                  "%s Date(%s) == date" % (dom, fmt))
        pm = proto["domains"][dom]
        dates = pd.to_datetime(df[pm["field_map"]["date"]], format="%Y-%m-%d")
        if dom != "Economy":
            check(list(df["orig_row"].astype(int)) == list(range(n)),
                  "%s orig_row identity" % dom)
        check(int((df["fact"].str.strip() == "").sum()) == 0,
              "%s empty fact remains" % dom)
        if dom != "Economy":
            raw = pd.read_csv(os.path.join(RAW_DIR, dom + ".csv"),
                              dtype=str, keep_default_na=False)
            empt = int((raw["fact"].str.strip() == "").sum())
            check(int((df["fact"] == NO_INFO).sum()) == empt,
                  "%s NO_INFO count %d != raw empty %d"
                  % (dom, int((df["fact"] == NO_INFO).sum()), empt))
        check(int((df["prior_history_avg"].str.strip() == "").sum()) == 0,
              "%s empty prior remains" % dom)

        # scaler exactness: parse OT strings with python float (strtod), refit
        ot = np.array([float(v) for v in df["OT"]], dtype=np.float64)
        check(float(scaler[dom]["mean"]) == float(ot[:n_train].mean())
              and float(scaler[dom]["std"]) == float(ot[:n_train].std(ddof=0)),
              "%s scaler mean/std exact recompute" % dom)

        for H, (e_tr, e_va, e_te) in sorted(exp_h.items()):
            got = s["window_counts"][str(H)]
            check(tuple(got[k] for k in ("train", "val", "test"))
                  == (e_tr, e_va, e_te),
                  "%s H%d counts %s != %s" % (dom, H, got, (e_tr, e_va, e_te)))
            w = pd.read_csv(os.path.join(OUT_DIR, "windows",
                                         "%s_H%d.csv" % (dom, H)))
            check(len(w) == e_tr + e_va + e_te,
                  "%s H%d window rows %d != %d" % (dom, H, len(w), e_tr + e_va + e_te))
            seg = {"train": (L, n_train - H, n_train),
                   "val": (n_train, val_end - H, val_end),
                   "test": (val_end, n - H, n)}
            for split, (lo, hi, seg_end) in seg.items():
                ws = w[w["split"] == split]
                check(len(ws) == hi - lo + 1,
                      "%s H%d %s count %d != %d" % (dom, H, split, len(ws), hi - lo + 1))
                svals = ws["s"].astype(int).tolist()
                check(svals == list(range(lo, hi + 1)),
                      "%s H%d %s s-sequence not [%d..%d]" % (dom, H, split, lo, hi))
                if len(ws):
                    check(int(ws["s"].max()) + H <= seg_end,
                          "%s H%d %s labels cross segment end" % (dom, H, split))
                ld = ws["label_dates"].map(json.loads)
                check(all(len(x) == H for x in ld),
                      "%s H%d %s label_dates length" % (dom, H, split))
                ok_ld = all(x == [str(dates.iloc[a + k].date()) for k in range(H)]
                            for x, a in zip(ld, svals))
                check(ok_ld, "%s H%d %s label_dates content" % (dom, H, split))
                idr = ws["input_date_range"].map(json.loads)
                ok_id = all(x == [str(dates.iloc[a - L].date()),
                                  str(dates.iloc[a - 1].date())]
                            for x, a in zip(idr, svals))
                check(ok_id, "%s H%d %s input_date_range content" % (dom, H, split))
                orow = ws["orig_row"].astype(int).tolist()
                check(all(o == int(df["orig_row"].iloc[a]) for o, a in zip(orow, svals)),
                      "%s H%d %s orig_row match" % (dom, H, split))
                if split == "test":
                    total_test += len(ws)

    check(total_test == 7388, "total test windows %d != 7388" % total_test)

    # Economy deep checks
    raw = pd.read_csv(os.path.join(RAW_DIR, "Economy.csv"),
                      dtype=str, keep_default_na=False)
    fin = pd.read_csv(os.path.join(OUT_DIR, "Economy.csv"),
                      dtype=str, keep_default_na=False)
    edates = pd.to_datetime(raw["date"], format="%Y-%m-%d")
    inv = int((edates.diff().dt.days < 0).sum())
    check(inv == 35, "Economy raw inversions %d != 35" % inv)
    orows = fin["orig_row"].astype(int).tolist()
    check(sorted(orows) == list(range(len(fin))), "Economy orig_row permutation")
    for i, o in enumerate(orows):
        if not fin.iloc[i].drop("orig_row").equals(raw.iloc[o]):
            check(False, "Economy row %d content != raw row %d" % (i, o))
            break
    fdates = pd.to_datetime(fin["date"], format="%Y-%m-%d")
    check(fdates.is_monotonic_increasing, "Economy final ascending")
    for rel, want in FROZEN_SHA.items():
        check(sha256(os.path.join(OUT_DIR, rel)) == want, "sha %s" % rel)

    # Health deep checks
    hf = pd.read_csv(os.path.join(OUT_DIR, "Health.csv"),
                     dtype=str, keep_default_na=False)
    new_ts = pd.to_datetime(hf["date_fixed"], format="%Y-%m-%d")
    old_ts = pd.to_datetime(hf["date"], format="%Y-%m-%d")
    calc = pd.to_datetime(pd.Series(
        [mmwr_monday(int(y), int(w))
         for y, w in zip(hf["YEAR"].astype(int), hf["WEEK"].astype(int))]))
    check((calc.values == new_ts.values).all(), "Health date_fixed == MMWR formula")
    check(int((new_ts.values != old_ts.values).sum()) == 832,
          "Health changed rows")
    check(str(new_ts.iloc[0].date()) == "1997-09-29"
          and str(new_ts.iloc[-1].date()) == "2024-05-06", "Health fixed range")
    check(new_ts.nunique() == len(hf) and old_ts.nunique() == len(hf) - 5,
          "Health uniqueness 1389 new / 1384 old")
    check(set(new_ts.diff().dt.days.dropna().astype(int)) == {7},
          "Health 7-day spacing")
    check(mmwr_monday(1998, 1).isoformat() == "1998-01-05"
          and mmwr_monday(2021, 1).isoformat() == "2021-01-04", "Health W01 checkpoints")
    fmap = pd.read_csv(os.path.join(OUT_DIR, "health_date_fix_map.csv"),
                       dtype=str, keep_default_na=False)
    check(len(fmap) == len(hf)
          and (fmap["new_date"].values == hf["date_fixed"].values).all()
          and (fmap["old_date"].values == hf["date"].values).all()
          and (fmap["orig_row"].astype(int).values == list(range(len(hf)))).all(),
          "Health fix map integrity")

    # freq / timef_dim recompute
    sys.path.insert(0, REPO)
    from utils.timefeatures import time_features_from_frequency_str
    for dom, s in splits.items():
        check(len(time_features_from_frequency_str(s["freq"])) == s["timef_dim"],
              "%s timef_dim" % dom)
        check(s["freq"] != "h", "%s freq must not be h" % dom)

    # protocol.json file hashes
    for rel, want in proto["files_sha256"].items():
        got = sha256(os.path.join(OUT_DIR, rel))
        check(got == want, "protocol hash %s" % rel)

    print("\nchecks run: %d, failures: %d" % (checks, len(fails)))
    rep = {"checks": checks, "failures": fails,
           "total_test_windows": total_test}
    with open(os.path.join(OUT_DIR, "window_check_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    if fails:
        sys.exit(2)
    print("WINDOW CHECK: ALL PASS (total test windows = %d)" % total_test)


if __name__ == "__main__":
    main()
