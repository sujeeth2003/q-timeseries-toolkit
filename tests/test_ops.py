import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qtk import ops  # noqa: E402


def data(nt, nq, ns, seed):
    rng = np.random.default_rng(seed)
    q_time = np.sort(rng.integers(0, 5000, nq)); q_sym = rng.integers(0, ns, nq); q_val = rng.random(nq)
    t_time = np.sort(rng.integers(0, 5000, nt)); t_sym = rng.integers(0, ns, nt)
    return t_sym, t_time, q_sym, q_time, q_val


class OpsTests(unittest.TestCase):
    def test_aj_variants_agree_including_ties_and_misses(self):
        for seed in range(6):
            d = data(120, 400, 8, seed)
            a, b, c = ops.aj_naive(*d), ops.aj_sorted(*d), ops.aj_numpy(*d)
            for x, y in ((a, b), (a, c)):
                np.testing.assert_allclose(np.nan_to_num(x, nan=-1), np.nan_to_num(y, nan=-1))

    def test_aj_known_answer(self):
        # quotes: sym0 @t10=1.0, sym0 @t20=2.0, sym1 @t15=9.0 ; trades: sym0@t19, sym0@t20 (equal time counts), sym0@t5 (none), sym1@t14 (none)
        r = ops.aj_numpy(np.array([0, 0, 0, 1]), np.array([19, 20, 5, 14]), np.array([0, 0, 1]), np.array([10, 20, 15]), np.array([1.0, 2.0, 9.0]))
        np.testing.assert_allclose(r[:2], [1.0, 2.0]); self.assertTrue(np.isnan(r[2]) and np.isnan(r[3]))

    def test_mavg_variants_agree_and_growing_window(self):
        x = np.random.default_rng(0).random(500)
        np.testing.assert_allclose(ops.mavg_naive(x, 20), ops.mavg_cumsum(x, 20))
        self.assertAlmostEqual(ops.mavg_cumsum(np.array([2., 4., 6.]), 5)[2], 4.0)     # fewer than w samples: average what exists

    def test_bars_variants_agree_and_ohlc_semantics(self):
        rng = np.random.default_rng(1); n = 5000
        sym = rng.integers(0, 6, n); t = np.sort(rng.integers(0, 100_000, n)); p = 50 + rng.normal(0, 1, n); z = rng.integers(1, 100, n)
        a, b = ops.bars_naive(sym, t, p, z, 5000), ops.bars_numpy(sym, t, p, z, 5000)
        self.assertEqual(a.keys(), b.keys())
        for k in a: np.testing.assert_allclose(a[k], b[k])
        o, h, l, c, vwap, vol = a[next(iter(a))]
        self.assertTrue(l <= o <= h and l <= c <= h and l <= vwap <= h)

    def test_row_and_column_sums_agree(self):
        col = np.random.default_rng(2).random(1000)
        self.assertAlmostEqual(ops.sum_rowstore([{"price": v} for v in col]), ops.sum_colstore(col))


if __name__ == "__main__":
    unittest.main()
