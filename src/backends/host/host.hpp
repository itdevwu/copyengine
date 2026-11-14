#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "backends/executor.hpp"
#include "core/convert.hpp"

namespace ce {

using ConvertFn = void (*)(const void* src, void* dst, int64_t n, const cv::Fn& fn);
// Element loop for (src dtype, dst dtype, identity?) — identity same-dtype is memcpy.
ConvertFn convert_fn(DType src, DType dst, bool identity);
cv::Fn to_cv(const Elementwise& f);

// Copies n elements between two host addresses, applying fn.
void host_copy(const void* src, DType st, void* dst, DType dt, int64_t n, const Elementwise& fn);

class ThreadPool {
 public:
  explicit ThreadPool(int threads);
  ~ThreadPool();
  int size() const { return static_cast<int>(workers_.size()) + 1; }
  // Runs fn(i) for i in [0, n); the caller participates. Not reentrant.
  void parallel_for(int64_t n, const std::function<void(int64_t)>& fn);

 private:
  void worker();
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(int64_t)>* job_ = nullptr;
  int64_t n_ = 0, next_ = 0, finished_ = 0;
  uint64_t gen_ = 0;
  bool stop_ = false;
};

class HostExecutor final : public Executor {
 public:
  explicit HostExecutor(int threads);
  std::vector<Strategy> candidates(const Group& g) override;
  int64_t run(Strategy s, const Group& g, const std::vector<Op>& ops, uintptr_t stream) override;
  void sync(std::span<const std::pair<uintptr_t, int>>) override {}
  bool async() const override { return false; }

 private:
  ThreadPool pool_;
};

}  // namespace ce
