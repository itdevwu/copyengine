// Footprints against brute force: affine_bytes must contain every byte of its
// set (dependencies rely on it: a missing byte is a race), equal the set when
// it is exact, and never exceed the hull.
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "core/plan.hpp"

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
      return 1;                                                              \
    }                                                                        \
  } while (0)

using ce::i64;

static std::set<i64> brute(i64 base, const std::vector<i64>& shape, const std::vector<i64>& stride, i64 width) {
  std::set<i64> b;
  std::vector<i64> i(shape.size(), 0);
  for (i64 n : shape)
    if (n <= 0) return b;
  for (;;) {
    i64 a = base;
    for (size_t k = 0; k < shape.size(); ++k) a += i[k] * stride[k];
    for (i64 w = 0; w < width; ++w) b.insert(a + w);
    size_t k = shape.size();
    while (k > 0 && ++i[k - 1] == shape[k - 1]) i[--k] = 0;
    if (k == 0) return b;
  }
}

static std::set<i64> bytes_of(const ce::Intervals& iv) {
  std::set<i64> b;
  for (size_t k = 0; k < iv.start.size(); ++k)
    for (i64 a = iv.start[k]; a < iv.end[k]; ++a) b.insert(a);
  return b;
}

static bool well_formed(const ce::Intervals& iv) {
  for (size_t k = 0; k < iv.start.size(); ++k) {
    if (iv.start[k] >= iv.end[k]) return false;
    if (k && iv.start[k] <= iv.end[k - 1]) return false;  // disjoint, sorted, merged
  }
  return true;
}

int main() {
  std::mt19937_64 rng(20261117);
  auto pick = [&](i64 lo, i64 hi) { return std::uniform_int_distribution<i64>(lo, hi)(rng); };
  int exact = 0, hulls = 0;
  for (int trial = 0; trial < 20000; ++trial) {
    const int rank = static_cast<int>(pick(0, 4));
    std::vector<i64> shape(rank), stride(rank);
    for (int k = 0; k < rank; ++k) {
      shape[k] = pick(1, 5);
      // Contiguous, gapped, overlapping, zero and negative steps.
      stride[k] = pick(0, 3) == 0 ? pick(-24, 24) : pick(1, 4) * (1 << pick(0, 4));
    }
    const i64 width = pick(1, 9), base = pick(0, 64) + 200;
    const size_t cap = pick(0, 3) == 0 ? static_cast<size_t>(pick(1, 4)) : ce::kFootprintCap;
    const ce::Intervals iv = ce::affine_bytes(base, shape, stride, width, cap);
    const std::set<i64> want = brute(base, shape, stride, width), got = bytes_of(iv);
    CHECK(well_formed(iv));
    for (i64 b : want) CHECK(got.count(b));  // sound: every byte touched is in the footprint
    if (!want.empty()) {
      CHECK(iv.lo() == *want.begin() && iv.hi() == *want.rbegin() + 1);  // never beyond the hull
    } else {
      CHECK(iv.empty());
    }
    if (got == want) {
      ++exact;
    } else {
      ++hulls;
      CHECK(iv.start.size() == 1);  // inexact only as the hull, past the cap
    }
    if (cap == ce::kFootprintCap) CHECK(got == want);  // small sets are exact at the default cap
  }
  // The case that motivated it: column slices of a block-major pool [blocks, row]
  // do not overlap one another, though their hulls do.
  {
    const i64 blocks = 64, row = 4096;
    const i64 shape[2] = {blocks, 512}, stride[2] = {row, 1};
    const ce::Intervals a = ce::affine_bytes(0, shape, stride, 1), b = ce::affine_bytes(512, shape, stride, 1);
    CHECK(a.start.size() == blocks && !ce::intersects(a, b));
    CHECK(ce::intersects(a, ce::affine_bytes(511, shape, stride, 1)));
  }
  std::printf("footprints: %d exact, %d hulls\n", exact, hulls);
  return 0;
}
