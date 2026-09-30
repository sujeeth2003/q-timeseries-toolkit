# kdb+/q Time-Series Toolkit (a learning project)

I used kdb+/q to understand *why* time-series databases are fast, then rebuilt its core operators myself, each at three or four speed levels, to see exactly where the speed comes from and how far it can be pushed. Every fast version is tested to return the same answer as the naive one.

> **kdb+/q is not installed here, so nothing in this repo runs q.** The `q:` lines in the source are the q equivalents I am reimplementing, shown for reference and **not executed**. Everything below is Python/numpy and C++ that I ran.

## The operators
| q | What it does | Versions here |
|---|---|---|
| `aj[`sym`time; trades; quotes]` | **as-of join**: each trade gets the latest quote at or before it, same symbol | naive scan, per-symbol binary search, one vectorised `searchsorted` on a composite key, C++ (binary search and a merge pass) |
| `xbar` + `select ... by sym, bar` | time bucketing with OHLC, VWAP (`wavg`) and volume per bucket | dict loop vs numpy sort + segment reductions |
| `mavg[w; x]` | moving average | re-sum each window vs cumulative sums vs running sum in C++ |
| columnar tables | why a column is one contiguous array | list-of-dicts vs a `float64` array |

## Results (this Windows laptop, single runs)
```
aj:    300 trades x 3,000 quotes : naive 131.5 ms | sorted+bisect 1.19 ms | numpy 0.102 ms   (1,288x vs naive)
       1M trades x 2M quotes     : numpy 278 ms  (278 ns/trade)
mavg:  20,000 rows : naive 41.1 ms | cumsum 0.277 ms  (148x);   5M rows: cumsum 80 ms
bars:  200,000 trades: dict loop 148 ms | numpy segments 19 ms   (7.6x)
rows vs columns: sum of 500,000 prices: list of dicts 37.5 ms | float64 column 0.154 ms   (243x)

C++ (2M trades x 4M quotes, 500 symbols):
       aj binary search (incl. sort) 425 ns/trade | aj merge pass 11.3 ns/trade  (37.7x)   results identical
       mavg(20) running sum 2.1 ns/row
```

## What I learned
1. **Exploit sorted order.** The as-of join is a merge of two time-sorted streams. Scanning is O(n x m); binary search is O(n log m); one merge pass with a "latest quote per symbol" table is **O(n + m) with sequential memory access**, which is why the C++ merge is 38x faster than binary search even though both are "fast" algorithms on paper: the search jumps around 4M keys (cache misses), the merge streams through memory.
2. **Fold the grouping into the key.** Doing the join per symbol needs a Python loop over symbols. Encoding `sym * SPAN + time` into one sortable integer lets a single `searchsorted` do every symbol at once, which is the vectorised trick q's `aj` gets from its attributes.
3. **Columnar layout is the foundation.** A column of floats is one contiguous array: summing it is a sequential, SIMD-friendly pass (243x faster than a list of dicts, whose values are scattered boxed objects).
4. **Watch for hidden quadratic work.** My first vectorised OHLC was *slower* than the loop (0.3x): I recomputed `reduceat` inside a per-bar loop, turning O(n) into O(bars x n). Hoisting it out made it 7.6x *faster*. Vectorising is not automatically fast; where the loop sits matters.
5. **Sliding windows: don't re-sum.** `mavg` by re-summing is O(n x w); a running sum is O(n) at ~2 ns/row.

## Run
```bash
pip install numpy
python -m unittest discover -s tests          # 5 tests: all variants agree (incl. ties, misses, equal timestamps), OHLC semantics
python bench.py
g++ -O2 -std=c++17 cpp/asof.cpp -o asof && ./asof [trades] [quotes] [symbols]
```

## Not covered
Real q/kdb+ features such as splayed/partitioned on-disk tables, attributes (`` `s# ``, `` `g# ``), IPC, and q-SQL are not reproduced. This is about the algorithms, not a database. A cross-check against real kdb+ output would be the next step once a licence/free-tier `q` is available.
