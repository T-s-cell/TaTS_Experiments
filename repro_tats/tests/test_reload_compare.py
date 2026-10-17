"""Unit tests for the reload comparator (baseline vs reload, no self-compare).

Regression guard for the audit finding: the reload phase previously
overwrote test_arrays.npz before comparing, so any diff collapsed to 0.

  python -m unittest repro_tats.tests.test_reload_compare -v
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from repro_tats.run_group import compare_reload, save_test_arrays


def make_arrays(N=7, H=6, seed=0):
    rng = np.random.RandomState(seed)
    arrays = (rng.randn(N, H, 1), rng.randn(N, H, 1),
              rng.randn(N, H, 1), rng.randn(N, H, 1), np.arange(N))
    identity = {
        "s": (np.arange(N) + 37).astype(np.int64),
        "orig_row": (np.arange(N) + 11).astype(np.int64),
        "label_dates": np.array(
            [["2021-%02d-%02d" % (m + 1, d + 1) for d in range(H)]
             for m in range(N)], dtype="U10"),
    }
    return arrays, identity


class TestCompareReload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tats_cmp_")
        self.arrays, self.identity = make_arrays()
        save_test_arrays(Path(self.tmp) / "test_arrays.npz",
                         self.arrays, self.identity)

    def test_identical_passes(self):
        v = compare_reload(self.tmp, self.arrays, self.identity, 1e-6)
        self.assertTrue(v["pass"])
        self.assertTrue(v["identity_pass"])
        self.assertEqual(v["max_abs_diff"], 0.0)

    def test_perturbed_prediction_fails(self):
        bad = (self.arrays[0] + 1e-3,) + self.arrays[1:]
        v = compare_reload(self.tmp, bad, self.identity, 1e-6)
        self.assertFalse(v["pass"])
        self.assertGreaterEqual(v["max_abs_diff"], 1e-3)

    def test_subtol_noise_passes(self):
        bad = (self.arrays[0] + 1e-9,) + self.arrays[1:]
        v = compare_reload(self.tmp, bad, self.identity, 1e-6)
        self.assertTrue(v["pass"])

    def test_identity_swap_fails(self):
        ident = dict(self.identity)
        ident["s"] = ident["s"][::-1].copy()
        v = compare_reload(self.tmp, self.arrays, ident, 1e-6)
        self.assertFalse(v["pass"])
        self.assertFalse(v["identity_pass"])

    def test_label_dates_swap_fails(self):
        ident = dict(self.identity)
        ident["label_dates"] = ident["label_dates"][::-1].copy()
        v = compare_reload(self.tmp, self.arrays, ident, 1e-6)
        self.assertFalse(v["pass"])
        self.assertFalse(v["identity_pass"])

    def test_window_count_mismatch_fails(self):
        bad = tuple(a[:-1] for a in self.arrays)
        ident = {k: v[:-1] for k, v in self.identity.items()}
        v = compare_reload(self.tmp, bad, ident, 1e-6)
        self.assertFalse(v["pass"])
        self.assertIn("error", v)

    def test_baseline_not_touched_by_compare(self):
        before = open(os.path.join(self.tmp, "test_arrays.npz"), "rb").read()
        bad = (self.arrays[0] + 1.0,) + self.arrays[1:]
        compare_reload(self.tmp, bad, self.identity, 1e-6)
        after = open(os.path.join(self.tmp, "test_arrays.npz"), "rb").read()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
