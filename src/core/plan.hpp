#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "copyengine/engine.hpp"
#include "copyengine/ir.hpp"
#include "copyengine/types.hpp"

namespace ce {

// 128-bit content hash; collisions are treated as impossible.
struct Hash128 {
  uint64_t a = 0, b = 0;
  bool operator==(const Hash128&) const = default;
};
struct Hash128Hasher {
  size_t operator()(const Hash128& h) const { return static_cast<size_t>(h.a ^ (h.b * 0x9E3779B97F4A7C15ull)); }
};

class Hasher {
 public:
  void add(uint64_t v);
  void add(std::span<const i64> v);
  void add(std::string_view s);
  Hash128 done() const { return {a_, b_}; }

 private:
  uint64_t a_ = 0x243F6A8885A308D3ull, b_ = 0x13198A2E03707344ull;
};

// The side of a copy as the planner sees it: offsets are absolute element
// indices (address / itemsize), so any two ops can be compared exactly.
struct Side {
  Device device;
  DType dtype = DType::u8;
  bool pinned = false;
  bool operator==(const Side&) const = default;
};

struct Op {
  Side src, dst;
  Runs runs;  // absolute element indices
  Elementwise fn;
  std::string ctx;
  int64_t src_lo = 0, src_hi = 0;  // the source allocation, absolute bytes: reads may widen within it
  Hash128 fp;  // content fingerprint
  // byte-interval read / write sets (absolute addresses), computed lazily
  mutable std::unique_ptr<Intervals> rd, wr;
  const Intervals& read() const;
  const Intervals& write() const;
};

Hash128 fingerprint(const Op& op);

// Rejects ops whose result would depend on write order or need memmove.
void check_safe(const Op& op);

// Dependency levels: ops in the same level have no conflicts and may be reordered and merged.
std::vector<int> waves(const std::vector<Op>& ops);

struct Group {
  Side src, dst;
  Elementwise fn;
  std::string ctx;
  std::vector<int> ops;  // indices into the flushed queue (original order)
  Runs runs;             // merged, coalesced
  Rects rects;           // structure recovered from runs
  Intervals src_extent;  // union of the source allocations (bytes): gap filling stays inside
  int wave = 0;
  // filled at plan time
  std::vector<Strategy> candidates;
  std::string bucket;
  // Backend-built tables (chunk lists, copy lists) per strategy, reused on
  // every plan-cache hit and frozen replay so the hot path does no planning.
  mutable std::array<std::shared_ptr<void>, kNumStrategies> exec_cache;

  bool pure() const { return fn.identity() && src.dtype == dst.dtype; }
  int64_t bytes() const { return runs.total() * static_cast<int64_t>(itemsize(src.dtype)); }
};

struct Plan {
  int waves = 0;
  std::vector<Group> groups;  // in wave order
};

Plan build_plan(const std::vector<Op>& ops);

std::string bucket_key(const Group& g, int nops);

}  // namespace ce
