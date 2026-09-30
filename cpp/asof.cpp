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

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

struct Rng { uint64_t s; uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; } };

int main(int argc, char** argv) {
  size_t nt = argc > 1 ? std::strtoull(argv[1], 0, 10) : 2000000, nq = argc > 2 ? std::strtoull(argv[2], 0, 10) : 4000000;
  uint32_t ns = argc > 3 ? (uint32_t)std::atoi(argv[3]) : 500;
  Rng r{88172645463325252ull};

  // both streams are time-sorted (as in a real tick database), interleaved in time
  struct Ev { int64_t t; uint32_t sym; double v; };
  std::vector<Ev> q(nq), tr(nt);
  int64_t t = 0; for (auto& e : q) { t += 1 + r.next() % 5; e = {t, (uint32_t)(r.next() % ns), (double)(r.next() % 100000) / 100}; }
  int64_t tmax = t; t = 0;
  for (auto& e : tr) { t += 1 + r.next() % (2 * tmax / nt + 1); e = {std::min(t, tmax), (uint32_t)(r.next() % ns), 0}; }

  // ---- variant 1: binary search on composite key ---------------------------------------------
  auto t0 = clk::now();
  std::vector<uint64_t> key(nq); std::vector<double> val(nq);
  { std::vector<size_t> ord(nq); for (size_t i = 0; i < nq; ++i) ord[i] = i;
    std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return q[a].sym != q[b].sym ? q[a].sym < q[b].sym : q[a].t < q[b].t; });
    for (size_t i = 0; i < nq; ++i) { key[i] = (uint64_t)q[ord[i]].sym << 40 | (uint64_t)q[ord[i]].t; val[i] = q[ord[i]].v; } }
  std::vector<double> res1(nt);
  for (size_t i = 0; i < nt; ++i) {
    uint64_t k = (uint64_t)tr[i].sym << 40 | (uint64_t)tr[i].t;
    size_t j = std::upper_bound(key.begin(), key.end(), k) - key.begin();
    res1[i] = (j > 0 && (key[j - 1] >> 40) == tr[i].sym) ? val[j - 1] : -1;
  }
  double d1 = ms_since(t0);

  // ---- variant 2: one merge pass, latest quote per symbol -------------------------------------
  t0 = clk::now();
  std::vector<double> last(ns, -1), res2(nt);
  size_t qi = 0;
  for (size_t i = 0; i < nt; ++i) {
    while (qi < nq && q[qi].t <= tr[i].t) { last[q[qi].sym] = q[qi].v; ++qi; }
    res2[i] = last[tr[i].sym];
  }
  double d2 = ms_since(t0);

  size_t bad = 0; for (size_t i = 0; i < nt; ++i) bad += res1[i] != res2[i];
  std::printf("%zu trades x %zu quotes, %u symbols\n", nt, nq, ns);
  std::printf("aj binary search (incl. sort) : %8.1f ms  %6.1f ns/trade\n", d1, d1 * 1e6 / nt);
  std::printf("aj merge pass                 : %8.1f ms  %6.1f ns/trade   (%.1fx faster)\n", d2, d2 * 1e6 / nt, d1 / d2);
  std::printf("results identical: %s\n", bad ? "NO" : "yes");

