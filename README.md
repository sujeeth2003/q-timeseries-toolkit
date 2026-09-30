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

