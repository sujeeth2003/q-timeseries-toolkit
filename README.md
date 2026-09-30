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

