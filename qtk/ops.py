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


def aj_sorted(t_sym, t_time, q_sym, q_time, q_val):
    """Group quotes per symbol once (sorted by time), then binary-search each trade: O(n log m)."""
    from bisect import bisect_right
    by_sym = {}
    for s, t, v in zip(q_sym, q_time, q_val):
        by_sym.setdefault(s, ([], []))[0].append(t); by_sym[s][1].append(v)
    out = np.full(len(t_time), np.nan)
    for i, (s, t) in enumerate(zip(t_sym, t_time)):
        if s in by_sym:
            times, vals = by_sym[s]
            k = bisect_right(times, t) - 1
            if k >= 0: out[i] = vals[k]
    return out


def aj_numpy(t_sym, t_time, q_sym, q_time, q_val):
    """Fully vectorised: fold the symbol into the sort key (sym * SPAN + time) so ONE np.searchsorted answers every lookup.
    A hit only counts if it landed on a quote of the same symbol."""
    span = int(max(q_time.max(), t_time.max())) + 1
    qkey = q_sym.astype(np.int64) * span + q_time
    order = np.argsort(qkey, kind="stable")
    qkey, qs, qv = qkey[order], q_sym[order], q_val[order]
    idx = np.searchsorted(qkey, t_sym.astype(np.int64) * span + t_time, side="right") - 1
    ok = (idx >= 0) & (qs[np.maximum(idx, 0)] == t_sym)
    return np.where(ok, qv[np.maximum(idx, 0)], np.nan)


# ------------------------------------------------------------------------------- xbar + OHLC + VWAP
# q:  select o:first price, h:max price, l:min price, c:last price, vwap:size wavg price by sym, bar:xbar[bucket;time] from trades
def bars_naive(sym, time, price, size, bucket):
    """dict of lists, Python loop. Returns {(sym, bar): (open, high, low, close, vwap, volume)} for trades in time order."""
    acc = {}
    for s, t, p, z in zip(sym, time, price, size):
        k = (int(s), int(t // bucket * bucket))
        if k not in acc: acc[k] = [p, p, p, p, p * z, z]
        else:
            a = acc[k]; a[1] = max(a[1], p); a[2] = min(a[2], p); a[3] = p; a[4] += p * z; a[5] += z
    return {k: (a[0], a[1], a[2], a[3], a[4] / a[5], a[5]) for k, a in acc.items()}


def bars_numpy(sym, time, price, size, bucket):
    """Sort by (sym, bar) once (stable, so time order is kept inside a bar), then segment reductions with reduceat/bincount."""
    bar = time // bucket * bucket
    order = np.lexsort((bar, sym))                      # primary key sym, then bar; stable
    s, b, p, z = sym[order], bar[order], price[order], size[order]
    new = np.r_[True, (s[1:] != s[:-1]) | (b[1:] != b[:-1])]
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], len(s)] - 1
    seg = np.cumsum(new) - 1
    vol = np.bincount(seg, z); pv = np.bincount(seg, p * z)
    hi, lo = np.maximum.reduceat(p, starts), np.minimum.reduceat(p, starts)      # once per column, not once per bar
    o, c, vw = p[starts], p[ends], pv / vol
    return {(int(s[a]), int(b[a])): (o[k], hi[k], lo[k], c[k], vw[k], vol[k]) for k, a in enumerate(starts)}


# ------------------------------------------------------------------------------- mavg
# q:  mavg[w; price]   (q averages over the samples available at the start, like the growing-window version here)
def mavg_naive(x, w):
    """O(n*w): re-sum the window at every step."""
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = sum(x[lo:i + 1]) / (i + 1 - lo)
    return out


def mavg_cumsum(x, w):
    """O(n): running sum via cumulative sums; window sum = c[i] - c[i-w]."""
    c = np.cumsum(x, dtype=np.float64)
    n = np.arange(1, len(x) + 1)
    win = np.minimum(n, w)
    s = c.copy(); s[w:] -= c[:-w]
    return s / win


# ------------------------------------------------------------------------------- row store vs column store
