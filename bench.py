"""Benchmark each operator from naive to fast:  python bench.py"""
import time

import numpy as np

from qtk import ops


def timed(f, *a, repeat=1):
    best = 1e9
    for _ in range(repeat):
        t = time.perf_counter(); r = f(*a); best = min(best, time.perf_counter() - t)
    return best, r


def make(nt, nq, ns, seed=0):
    rng = np.random.default_rng(seed)
    q_time = np.sort(rng.integers(0, 10_000_000, nq)); q_sym = rng.integers(0, ns, nq); q_val = rng.random(nq) * 100
    t_time = np.sort(rng.integers(0, 10_000_000, nt)); t_sym = rng.integers(0, ns, nt)
    return t_sym, t_time, q_sym, q_time, q_val


def main():
    print("== aj (as-of join) ==")
    small = make(300, 3000, 20)
    tn, a = timed(ops.aj_naive, *small)
    ts, b = timed(ops.aj_sorted, *small)
    tp, c = timed(ops.aj_numpy, *small, repeat=5)
    assert np.allclose(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1)) and np.allclose(np.nan_to_num(a, nan=-1), np.nan_to_num(c, nan=-1))
    print(f"  300 trades x 3,000 quotes : naive {tn * 1e3:8.1f} ms | sorted+bisect {ts * 1e3:6.2f} ms | numpy {tp * 1e3:6.3f} ms   (numpy {tn / tp:,.0f}x faster than naive)")
    big = make(1_000_000, 2_000_000, 500)
    tp, _ = timed(ops.aj_numpy, *big, repeat=3)
    print(f"  1M trades x 2M quotes     : numpy {tp * 1e3:8.0f} ms  ({tp * 1e9 / 1e6:.0f} ns/trade)")

    print("== mavg(20) ==")
    x = np.random.default_rng(1).random(20000)
    tn, a = timed(ops.mavg_naive, x, 20); tc, b = timed(ops.mavg_cumsum, x, 20, repeat=5)
    assert np.allclose(a, b)
    print(f"  20,000 rows : naive {tn * 1e3:7.1f} ms | cumsum {tc * 1e3:6.3f} ms   ({tn / tc:,.0f}x)")
    x = np.random.default_rng(1).random(5_000_000); tc, _ = timed(ops.mavg_cumsum, x, 20, repeat=3)
    print(f"  5M rows     : cumsum {tc * 1e3:7.1f} ms")

