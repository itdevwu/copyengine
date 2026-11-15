#include "backends/host/host.hpp"

#include <algorithm>
#include <cstring>

#include "copyengine/error.hpp"

namespace ce {

// ---- element conversion -----------------------------------------------------------

namespace {

template <DType T> struct ctype;
template <> struct ctype<DType::f64> { using type = double; };
template <> struct ctype<DType::f32> { using type = float; };
template <> struct ctype<DType::f16> { using type = cv::half_t; };
template <> struct ctype<DType::bf16> { using type = cv::bf16_t; };
template <> struct ctype<DType::i64> { using type = int64_t; };
template <> struct ctype<DType::i32> { using type = int32_t; };
template <> struct ctype<DType::i16> { using type = int16_t; };
template <> struct ctype<DType::i8> { using type = int8_t; };
template <> struct ctype<DType::u8> { using type = uint8_t; };
template <> struct ctype<DType::b8> { using type = cv::bool_t; };
template <> struct ctype<DType::f8e4m3> { using type = cv::f8e4m3_t; };
template <> struct ctype<DType::f8e5m2> { using type = cv::f8e5m2_t; };
template <> struct ctype<DType::f8e8m0> { using type = cv::f8e8m0_t; };

template <class D, class S>
void loop_cast(const void* src, void* dst, int64_t n, const cv::Fn&) {
  const S* s = static_cast<const S*>(src);
  D* d = static_cast<D*>(dst);
  for (int64_t i = 0; i < n; ++i) {
    S x;
    std::memcpy(&x, s + i, sizeof(S));  // tolerate unaligned host views
    D y = cv::cast<D>(x);
    std::memcpy(d + i, &y, sizeof(D));
  }
}

template <class D, class S>
void loop_apply(const void* src, void* dst, int64_t n, const cv::Fn& f) {
  const S* s = static_cast<const S*>(src);
  D* d = static_cast<D*>(dst);
  for (int64_t i = 0; i < n; ++i) {
    S x;
    std::memcpy(&x, s + i, sizeof(S));
    D y = cv::apply<D>(x, f);
    std::memcpy(d + i, &y, sizeof(D));
  }
}

template <int S, int D>
constexpr ConvertFn pick(bool identity) {
  using ST = typename ctype<static_cast<DType>(S)>::type;
  using DT = typename ctype<static_cast<DType>(D)>::type;
  return identity ? &loop_cast<DT, ST> : &loop_apply<DT, ST>;
}

struct Table {
  ConvertFn fn[kNumConvertible][kNumConvertible][2];  // f4x2 never converts
  template <int S, int D>
  void fill_one() {
    fn[S][D][0] = pick<S, D>(false);
    fn[S][D][1] = pick<S, D>(true);
  }
  template <int S, int... Ds>
  void fill_row(std::integer_sequence<int, Ds...>) {
    (fill_one<S, Ds>(), ...);
  }
  template <int... Ss>
  void fill(std::integer_sequence<int, Ss...>) {
    (fill_row<Ss>(std::make_integer_sequence<int, kNumConvertible>{}), ...);
  }
  Table() { fill(std::make_integer_sequence<int, kNumConvertible>{}); }
};

const Table& table() {
  static const Table t;
  return t;
}

}  // namespace

ConvertFn convert_fn(DType src, DType dst, bool identity) {
  return table().fn[static_cast<int>(src)][static_cast<int>(dst)][identity ? 1 : 0];
}

cv::Fn to_cv(const Elementwise& f) { return {f.scale, f.bias, f.lo, f.hi, f.round, f.clamp}; }

void host_copy(const void* src, DType st, void* dst, DType dt, int64_t n, const Elementwise& fn) {
  if (n <= 0) return;
  if (fn.identity() && st == dt) {
    std::memcpy(dst, src, static_cast<size_t>(n) * itemsize(st));
    return;
  }
  const cv::Fn f = to_cv(fn);
  convert_fn(st, dt, fn.identity())(src, dst, n, f);
}

// ---- thread pool ----------------------------------------------------------------------

ThreadPool::ThreadPool(int threads) {
  for (int i = 1; i < threads; ++i) workers_.emplace_back([this] { worker(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : workers_) t.join();
}

void ThreadPool::worker() {
  uint64_t seen = 0;
  std::unique_lock<std::mutex> lk(mu_);
  for (;;) {
    cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
    if (stop_) return;
    seen = gen_;
    while (next_ < n_) {
      const int64_t i = next_++;
      lk.unlock();
      (*job_)(i);
      lk.lock();
      if (++finished_ == n_) done_cv_.notify_all();
    }
  }
}

void ThreadPool::parallel_for(int64_t n, const std::function<void(int64_t)>& fn) {
  if (n <= 0) return;
  if (workers_.empty() || n == 1) {
    for (int64_t i = 0; i < n; ++i) fn(i);
    return;
  }
  std::unique_lock<std::mutex> lk(mu_);
  job_ = &fn;
  n_ = n, next_ = 0, finished_ = 0;
  ++gen_;
  cv_.notify_all();
  while (next_ < n_) {
    const int64_t i = next_++;
    lk.unlock();
    fn(i);
    lk.lock();
    ++finished_;
  }
  done_cv_.wait(lk, [&] { return finished_ == n_; });
  job_ = nullptr;
  n_ = 0;
}

// ---- executor -------------------------------------------------------------------------

namespace {
constexpr int64_t kParallelMinBytes = 1 << 20;
constexpr int64_t kTaskBytes = 256 << 10;

uintptr_t addr(i64 idx, DType t) { return static_cast<uintptr_t>(idx) * itemsize(t); }

void copy_rows(const Group& g, size_t r, i64 row_begin, i64 row_end) {
  const Rects& R = g.rects;
  for (i64 h = row_begin; h < row_end; ++h) {
    const i64 s = R.src[r] + h * R.src_pitch[r], d = R.dst[r] + h * R.dst_pitch[r];
    host_copy(reinterpret_cast<const void*>(addr(s, g.src.dtype)), g.src.dtype,
              reinterpret_cast<void*>(addr(d, g.dst.dtype)), g.dst.dtype, R.width[r], g.fn);
  }
}
}  // namespace

HostExecutor::HostExecutor(int threads) : pool_(std::max(1, threads)) {}

std::vector<Strategy> HostExecutor::candidates(const Group& g) {
  if (!g.ops.empty() && g.tiles) return {Strategy::tile};
  if (pool_.size() > 1 && g.bytes() >= kParallelMinBytes) return {Strategy::runs_parallel, Strategy::runs};
  return {Strategy::runs};
}

int64_t HostExecutor::run(Strategy s, const Group& g, const std::vector<Op>& ops, uintptr_t) {
  switch (s) {
    case Strategy::passthrough: {
      int64_t calls = 0;
      for (int k : g.ops) {
        const Op& op = ops[k];
        for (size_t i = 0; i < op.runs().size(); ++i) {
          host_copy(reinterpret_cast<const void*>(addr(op.runs().src[i], op.src.dtype)), op.src.dtype,
                    reinterpret_cast<void*>(addr(op.runs().dst[i], op.dst.dtype)), op.dst.dtype, op.runs().len[i],
                    op.fn);
        }
        calls += static_cast<int64_t>(op.runs().size());
      }
      return calls;
    }
    case Strategy::runs:
      for (size_t r = 0; r < g.rects.size(); ++r) copy_rows(g, r, 0, g.rects.height[r]);
      return static_cast<int64_t>(g.rects.size());
    case Strategy::runs_parallel: {
      // Tasks of ~kTaskBytes: whole rects, or row ranges of big rects. Long
      // single rows are split by element range.
      struct Task {
        size_t r;
        i64 row0, row1, col0, col1;
      };
      std::vector<Task> tasks;
      const i64 isz = static_cast<i64>(std::max(itemsize(g.src.dtype), itemsize(g.dst.dtype)));
      const i64 per = std::max<i64>(1, kTaskBytes / isz);
      const Rects& R = g.rects;
      for (size_t r = 0; r < R.size(); ++r) {
        if (R.width[r] >= per) {
          for (i64 h = 0; h < R.height[r]; ++h)
            for (i64 c = 0; c < R.width[r]; c += per) tasks.push_back({r, h, h + 1, c, std::min(R.width[r], c + per)});
        } else {
          const i64 rows = std::max<i64>(1, per / R.width[r]);
          for (i64 h = 0; h < R.height[r]; h += rows) tasks.push_back({r, h, std::min(R.height[r], h + rows), 0, R.width[r]});
        }
      }
      pool_.parallel_for(static_cast<int64_t>(tasks.size()), [&](int64_t i) {
        const Task& t = tasks[i];
        for (i64 h = t.row0; h < t.row1; ++h) {
          const i64 s = R.src[t.r] + h * R.src_pitch[t.r] + t.col0, d = R.dst[t.r] + h * R.dst_pitch[t.r] + t.col0;
          host_copy(reinterpret_cast<const void*>(addr(s, g.src.dtype)), g.src.dtype,
                    reinterpret_cast<void*>(addr(d, g.dst.dtype)), g.dst.dtype, t.col1 - t.col0, g.fn);
        }
      });
      return static_cast<int64_t>(tasks.size());
    }
    case Strategy::tile: {  // units in parallel: each touches only its own values, scale and zero point
      std::vector<std::pair<const cv::TileK*, long long>> parts;  // (op, first unit) in chunks
      constexpr long long kUnits = 256;
      for (int k : g.ops)
        for (long long u = 0; u < ops[k].tile->units(); u += kUnits) parts.emplace_back(ops[k].tile.get(), u);
      pool_.parallel_for(static_cast<int64_t>(parts.size()), [&](int64_t i) {
        const auto [t, u0] = parts[i];
        const long long u1 = std::min(t->units(), u0 + kUnits);
        for (long long u = u0; u < u1; ++u) cv::tile_unit(*t, u);
      });
      return static_cast<int64_t>(parts.size());
    }
    default:
      fail(Errc::unsupported, "host backend cannot run strategy " + std::string(strategy_name(s)));
  }
}

}  // namespace ce
