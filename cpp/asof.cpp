// The as-of join, xbar and moving average in C++, to see how far the algorithmic ideas go when the interpreter is gone.
//
//   asof_bench [n_trades=2000000] [n_quotes=4000000] [n_syms=500]
//
// aj variants (all must agree): binary search per trade (upper_bound on a composite sym|time key, the same trick as the
// numpy version) versus a single merge pass over both time-sorted streams keeping the latest quote per symbol (O(n+m),
// sequential memory access, no log factor). Prints ns per trade.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

