#pragma once

#include <cstdint>
#include <vector>

#include "copyengine/engine.hpp"
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
  // Blocks until everything issued on `stream` is done.
  virtual void sync(uintptr_t stream) = 0;
  // Whether run() may return before the copy is complete.
  virtual bool async() const = 0;
};

}  // namespace ce
