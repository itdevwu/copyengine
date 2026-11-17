#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "copyengine/engine.hpp"
#include "core/json.hpp"
#include "core/plan.hpp"

namespace ce {

// A backend turns a planned Group into platform calls.
class Executor {
 public:
  virtual ~Executor() = default;
  // Strategies worth trying, best prior first. The engine appends passthrough.
  virtual std::vector<Strategy> candidates(const Group& g) = 0;
  // Issues the copy on `stream`; returns the number of backend calls made.
  virtual int64_t run(Strategy s, const Group& g, const std::vector<Op>& ops, uintptr_t stream) = 0;
  // Blocks until everything issued on these streams is done.
  // Each (stream, device) the work ran on; the device names the context of a
  // legacy stream (0), even one this backend never set up itself (work it
  // left to the framework).
  virtual void sync(std::span<const std::pair<uintptr_t, int>> streams) = 0;
  // Whether run() may return before the copy is complete.
  virtual bool async() const = 0;
  // Timing without blocking: mark() records a point on `stream` of `device`'s
  // queue (-1: nothing to time, the backend is synchronous); elapsed() is the
  // microseconds between two marks once both have completed, else -1. Marks
  // are released by elapsed() returning a value, or by release().
  virtual int64_t mark(uintptr_t /*stream*/, int /*device*/) { return -1; }
  virtual double elapsed(int64_t /*a*/, int64_t /*b*/) { return 0; }
  virtual void release(int64_t /*m*/) {}
  // Work queued on `later` from now on runs after what is queued on `earlier`
  // (an event: the host does not wait).
  virtual void order(uintptr_t /*later*/, uintptr_t /*earlier*/, int /*device*/) {}
  // Whether the work before mark `m` has completed (does not release it).
  virtual bool ready(int64_t /*m*/) { return true; }
  // CUDA Graph capture. While `stream` is being captured, work may only be
  // enqueued on it: no allocation, synchronization or event query.
  virtual bool capturing(uintptr_t /*stream*/) { return false; }
  // Whether running `s` for `g` now only enqueues stream work (its tables
  // already resident, no staging, pinned or device memory only).
  virtual bool capture_safe(Strategy /*s*/, const Group& /*g*/) { return true; }
  // Admission results etc. for stats().
  virtual Json info() { return Json(true); }
};

}  // namespace ce
