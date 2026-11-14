// The runtime-dispatched conversions must equal the templated ones bit for bit,
// for every dtype pair, with and without a function.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "backends/host/host.hpp"
#include "core/convert.hpp"

int main() {
  using namespace ce;
  std::mt19937_64 rng(7);
  const int n = 4096;
  std::vector<unsigned char> src(n * 8), a(n * 8), b(n * 8);
  const cv::Fn fns[] = {{0.37f, -1.5f, 0, 0, false, false}, {8.0f, 0.25f, -100.0f, 100.0f, true, true}};
  for (int s = 0; s < kNumConvertible; ++s) {
    const DType st = static_cast<DType>(s);
    // random bits, then patch in values that matter for each source type
    for (auto& x : src) x = static_cast<unsigned char>(rng());
    if (st == DType::f32 || st == DType::f64) {
      const double specials[] = {0.0, -0.0, 1.5, 2.5, -2.5, 127.5, -128.5, 65504.0, 65520.0, 1e-8, 3e38, -3e38, 1e300,
                                 std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                                 9.3e18, -9.3e18, 2147483648.0, 4e9};
      int k = 0;
      for (double v : specials) {
        if (st == DType::f32) {
          float f = static_cast<float>(v);
          std::memcpy(&src[4 * k++], &f, 4);
        } else {
          std::memcpy(&src[8 * k++], &v, 8);
        }
      }
    }
    for (int d = 0; d < kNumConvertible; ++d) {
      const DType dt = static_cast<DType>(d);
      const size_t ss = itemsize(st), ds = itemsize(dt);
      if (s != d) {
        std::memset(a.data(), 0, a.size());
        std::memset(b.data(), 0, b.size());
        convert_fn(st, dt, true)(src.data(), a.data(), n, {});
        for (int i = 0; i < n; ++i) cv::dyn_cast(&src[i * ss], s, &b[i * ds], d);
        if (std::memcmp(a.data(), b.data(), n * ds) != 0) {
          std::fprintf(stderr, "dyn_cast differs for %d -> %d\n", s, d);
          return 1;
        }
      }
      for (const cv::Fn& f : fns) {
        std::memset(a.data(), 0, a.size());
        std::memset(b.data(), 0, b.size());
        convert_fn(st, dt, false)(src.data(), a.data(), n, f);
        for (int i = 0; i < n; ++i) cv::dyn_apply(&src[i * ss], s, &b[i * ds], d, f);
        if (std::memcmp(a.data(), b.data(), n * ds) != 0) {
          std::fprintf(stderr, "dyn_apply differs for %d -> %d\n", s, d);
          return 1;
        }
      }
    }
  }
  return 0;
}
