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


