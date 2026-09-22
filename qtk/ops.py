"""kdb+/q time-series primitives, each implemented several ways from naive to fast.

Study notes: q is fast because data is COLUMNAR (each column is one contiguous typed array), tables are usually sorted by
time (so lookups are binary searches, not scans), and operators work on whole vectors. Each function below has:
  *_naive    the obvious loop, as you would first write it
  *_sorted   the algorithmic improvement (exploit sorted order: one pass or binary search)
  *_numpy    whole-column vectorised version (what q does), no Python-level loop
All variants must return identical results (tests/test_ops.py).
"""
import numpy as np


# ------------------------------------------------------------------------------- aj: as-of join
# q:   aj[`sym`time; trades; quotes]   -> for each trade, the LAST quote with the same sym and quote.time <= trade.time
def aj_naive(t_sym, t_time, q_sym, q_time, q_val):
    """O(trades x quotes): scan every quote for every trade."""
    out = np.full(len(t_time), np.nan)
    for i in range(len(t_time)):
        best_t = -1
        for j in range(len(q_time)):
            if q_sym[j] == t_sym[i] and q_time[j] <= t_time[i] and q_time[j] >= best_t:
                best_t, out[i] = q_time[j], q_val[j]
    return out


