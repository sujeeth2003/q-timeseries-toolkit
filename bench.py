"""Benchmark each operator from naive to fast:  python bench.py"""
import time

import numpy as np

from qtk import ops


def timed(f, *a, repeat=1):
    best = 1e9
    for _ in range(repeat):
        t = time.perf_counter(); r = f(*a); best = min(best, time.perf_counter() - t)
    return best, r


