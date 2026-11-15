#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/tile.hpp"
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

// A table-free statement kept in structured form: absolute element indices in
// units of the original dtypes (also for pure copies, whose runs are bytes).
struct Form {
  static constexpr int kMaxRank = 6;
  int rank = 0;
  int64_t numel = 1;
  std::array<i64, kMaxRank> shape{}, ss{}, ds{};
  i64 s_base = 0, d_base = 0;
  DType sdt = DType::u8, ddt = DType::u8;
  // At most one dimension per side may read a device index table (its
  // pointer is per call: Op::stab / Op::dtab); -1: none.
  int stab_dim = -1, dtab_dim = -1;
  i64 stab_step = 1, dtab_step = 1, sbound = 0, dbound = 0;
  bool stab_host = false, dtab_host = false;  // the table is in host memory (DeviceTable::kHost)
  bool device_tables() const { return stab_dim >= 0 || dtab_dim >= 0; }
  bool host_tables() const { return (stab_dim >= 0 && stab_host) || (dtab_dim >= 0 && dtab_host); }
  // Lowest / highest element index touched on each side.
  std::pair<i64, i64> src_extent() const;
  std::pair<i64, i64> dst_extent() const;
  bool same_structure(const Form& o) const {
    return rank == o.rank && shape == o.shape && ss == o.ss && ds == o.ds && sdt == o.sdt && ddt == o.ddt &&
           stab_dim == o.stab_dim && dtab_dim == o.dtab_dim && stab_step == o.stab_step &&
           dtab_step == o.dtab_step && sbound == o.sbound && dbound == o.dbound && stab_host == o.stab_host &&
           dtab_host == o.dtab_host;
  }
};
// The runs a form with host tables reads now: `stab` / `dtab` are its tables'
// host pointers (int64, the form's steps). Absolute element indices; the
// innermost dimension stays one run where it is contiguous on both sides.
// An index outside its bound is an error.
Runs lower_form(const Form& f, const int64_t* stab, const int64_t* dtab);


// An op's runs and their lazily computed byte intervals. Shared by every
// queued copy of a cached statement, so a statement-cache hit copies nothing.
struct OpBody {
  Runs runs;  // absolute element indices
  mutable std::unique_ptr<Intervals> rd, wr;
};

struct Op {
  Side src, dst;
  std::shared_ptr<const OpBody> body;
  const Runs& runs() const;
  Elementwise fn;
  std::string ctx;
  int64_t src_lo = 0, src_hi = 0;  // the source allocation, absolute bytes: reads may widen within it
  int64_t dst_lo = 0, dst_hi = 0;  // the destination allocation, absolute bytes
  Hash128 key;                     // op class for interception-time decisions (sides, fn, size buckets)
  // per submission, not part of the cached op
  std::shared_ptr<NativeOp> native;  // the frontend's own implementation (the baseline), or null
  double enqueue_us = 0;           // host time spent submitting it
  bool host_async = false;         // Hints::host_async
  int64_t weight = 1;              // runs this op stands for in online sampling (see Impl::native_now)
  uintptr_t stab = 0, dtab = 0;    // device index tables of form->stab_dim / dtab_dim, this call's
  // A statement with device tables has no runs (its indices are not known on
  // the host): `form` is its only executable description, and its footprint
  // is its whole source / destination allocation.
  bool opaque = false;
  double bandwidth = 0;            // bytes/s limit of its scope (0: none), mixed into fp
  std::shared_ptr<const Form> form;  // null when the statement has tables or came in as runs
  // A tile op (quantization while copying): its kernel description, and the
  // footprints beyond src / dst (scale, zero point), each on its device.
  std::shared_ptr<const cv::TileK> tile;
  // Its host index tables (TileK::xi / ci point into them), and whether it
  // reads a GPU index table (this call's pointer: the kernel's description is
  // written per call).
  std::shared_ptr<const std::vector<int64_t>> tile_index[2];
  bool tile_dev_index = false;
  // Host index tables read when the op runs (TileK::xi / ci point at a step
  // plan's parameter buffer): their values are checked then, not when queued.
  bool tile_host_ptr[2] = {false, false};
  // Its host tensors are compact enough for device staging: what the kernel
  // reads spans at most twice the bytes it holds, what it writes is dense,
  // and no index tables (rows could be anywhere). Set when queued.
  bool tile_host = false, tile_staged_ok = false;
  struct Extra {
    Device device;
    int64_t lo = 0, hi = 0;  // absolute bytes
    bool write = false;
  };
  std::vector<Extra> extras;
  Hash128 fp;  // content fingerprint
  // byte-interval read / write sets (absolute addresses), computed lazily
  const Intervals& read() const;
  const Intervals& write() const;
};

Hash128 fingerprint(const Op& op);
// Sides, function and context only: the part of an op's class that is not size.
uint64_t fingerprint_sides(const Op& op);

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
  int64_t op_runs = 0;   // runs of the ops as submitted: what passthrough issues
  int64_t opaque_elems = 0;  // elements of ops without runs (device tables): only their forms can run them
  bool opaque() const { return opaque_elems > 0; }
  bool tiles = false;  // tile ops (quantization while copying): only the tile strategies run them
  bool tile_host_sides = false;  // some tensor of its tile ops is in host memory
  bool tile_staged_ok = false;   // ... and every op can go through device staging (Op::tile_staged_ok)
  Rects rects;           // structure recovered from runs
  // Allocation boundaries on each side, in run units: one copy call never
  // crosses them (separate allocations, even adjacent ones, are not one range
  // for the driver). Cuts strictly inside another described allocation are
  // dropped (the same memory described twice); runs never cross a cut.
  std::vector<i64> src_cuts, dst_cuts;
  // The source allocations, [lo, hi) bytes, not merged: gap filling stays in one.
  std::vector<std::pair<i64, i64>> src_allocs;
  std::vector<std::shared_ptr<const Form>> forms;  // one per op when every op is structured, else empty
  int wave = 0;
  double bandwidth = 0;  // its ops' limit: ops with different limits never share a group
  // filled at plan time
  std::vector<Strategy> candidates;
  std::string bucket;
  // Backend-built tables (chunk lists, copy lists) per strategy, reused on
  // every plan-cache hit and frozen replay so the hot path does no planning.
  mutable std::array<std::shared_ptr<void>, kNumStrategies> exec_cache;

  bool pure() const { return fn.identity() && src.dtype == dst.dtype; }
  // Cached: a plan's groups never change, and runs can be millions long.
  int64_t bytes() const {
    if (bytes_ < 0) bytes_ = (runs.total() + opaque_elems) * static_cast<int64_t>(itemsize(src.dtype));
    return bytes_;
  }

 private:
  mutable int64_t bytes_ = -1;
};

struct Plan {
  int waves = 0;
  std::vector<Group> groups;  // in wave order
};

Plan build_plan(const std::vector<Op>& ops);

std::string bucket_key(const Group& g, int nops);

}  // namespace ce
