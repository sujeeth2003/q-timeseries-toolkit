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


