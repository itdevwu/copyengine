#pragma once

#include <cstdint>
#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "copyengine/ir.hpp"

namespace ce {

// f8e4m3 / f8e5m2 / f8e8m0: torch's float8_e4m3fn / float8_e5m2 /
// float8_e8m0fnu, with torch's conversions (overflow to NaN, no saturation;
// clamp first to saturate). f4x2 is two packed fp4 values per byte: moved as
// bytes, never converted.
enum class DType : uint8_t { f64, f32, f16, bf16, i64, i32, i16, i8, u8, b8, f8e4m3, f8e5m2, f8e8m0, f4x2 };
inline constexpr int kNumDTypes = 14;
inline constexpr int kNumConvertible = 13;  // dtypes before f4x2 convert into each other

inline bool convertible(DType t) { return static_cast<int>(t) < kNumConvertible; }

size_t itemsize(DType t);
std::string_view dtype_name(DType t);
DType parse_dtype(std::string_view name);  // "float32", "bfloat16", "bool", ...

enum class DeviceKind : uint8_t { host, cuda };

struct Device {
  DeviceKind kind = DeviceKind::host;
  int index = 0;
  bool operator==(const Device&) const = default;
};

// One side of a copy: a typed view of an allocation. Offsets in requests are
// elements relative to `base`; [base, base + nbytes) bounds every access.
// Host memory as the driver sees it. DMA reaches pinned memory; kernels reach
// it at its own address only when it is mapped there (memory from cudaHostAlloc
// is; cudaHostRegister'ed memory may have another device address, e.g. WSL2).
enum class HostMemory : uint8_t { pageable, pinned, mapped };

struct Endpoint {
  Device device;
  uintptr_t base = 0;
  int64_t nbytes = 0;
  DType dtype = DType::u8;
  bool pinned = false;  // host only: page-locked (DMA reaches it)
  bool mapped = false;  // host only: pinned, and kernels reach it at `base` (see host_memory)
};

// y = clamp(round(float(x) * scale + bias)), then converted to the destination
// dtype. A closed set so it commutes with data movement.
//
// Conversion rules (host and device produce identical bits):
//   * float -> float rounds to nearest even; f64 -> f16/bf16 goes through f32.
//   * int -> float rounds to nearest even; int -> int wraps (two's complement).
//   * float -> int truncates toward zero, saturates, NaN -> 0.
//   * -> bool is (x != 0).
//   * non-identity functions compute in f32 without FMA contraction; round is
//     round-half-to-even.
struct Elementwise {
  float scale = 1.0f;
  float bias = 0.0f;
  bool round = false;
  bool clamp = false;
  float lo = 0.0f, hi = 0.0f;

  bool identity() const { return scale == 1.0f && bias == 0.0f && !round && !clamp; }
  bool operator==(const Elementwise&) const = default;
};

// One copy as the framework itself would run it (for torch: aten::copy_), the
// no-regression baseline. Implemented by each frontend.
class NativeOp {
 public:
  virtual ~NativeOp() = default;
  // Runs the copy on `stream`, ordered like any other work on it.
  virtual void run(uintptr_t stream) = 0;
  // The same, where the host must not wait for it even if the caller asked
  // for a blocking copy (an async scope): runs it as non_blocking.
  virtual void run(uintptr_t stream, bool host_async) {
    (void)host_async;
    run(stream);
  }
  // Whether running it touches host memory on the CPU when it is called (a
  // conversion, gather or pageable copy done by the host), rather than only in
  // stream order: the engine then lets pending GPU work on host memory finish
  // first. Unknown: true.
  virtual bool cpu_host_access() const { return true; }
};

// A tile op (quantization while copying, docs/design/v0.4.md): rows of `upr`
// units of `g` values along the innermost dimension of x. Each tensor is an
// endpoint plus its layout in bytes from the endpoint's base: `offset` of the
// first unit, `row_stride` of up to three row dimensions, `step` between
// consecutive values (x, q; packed formats: between bytes of two values) or
// units (scale, zero).
struct TileSpec {
  struct Side {
    Endpoint ep;
    int64_t offset = 0;
    std::array<int64_t, 3> row_stride{0, 0, 0};
    int64_t step = 0;
    int64_t pitch = 0;  // box units: bytes between a box's rows (x and q)
  };
  int fmt = 0;              // ce::cv::QFmt
  bool quantize = true;     // else dequantize
  bool static_scale = false;  // quantize with the given scale / zero point (read, not written)
  int g = 1;                // values per unit (per row of a box)
  // Box units: `bh` rows of `g` values each, `pitch` bytes apart in x and q
  // (one scale per box: block FP8 / INT8, as DeepSeek-V3 stores weights).
  int bh = 1;
  float eps = 1e-12f, gs = 1.0f;
  std::array<int64_t, 3> rows{1, 1, 1};
  int64_t upr = 1;          // units per row
  Side x, q, scale, zero;   // zero: uint4 only
  // Gathers / scatters: the first row dimension of x (x_index), or of q,
  // scale and zero (index), goes through an index table: int64 values in GPU
  // memory at `dev`, or given on the host; each below `bound` (the indexed
  // tensor's first dimension). rows[0] is the number of indices.
  struct Index {
    uintptr_t dev = 0;
    std::vector<int64_t> host;
    int64_t bound = 0;
    // Host values read when the copy runs, from rows[0] int64 at this address
    // (a step plan's parameter buffer, refilled between calls); 0: none.
    uintptr_t host_ptr = 0;
    // How many indices `dev` / `host_ptr` hold (host: host.size()); must be
    // rows[0], checked by the engine: the kernel reads rows[0] of them.
    int64_t count = -1;
    bool used() const { return dev != 0 || !host.empty() || host_ptr != 0; }
    int64_t size() const { return dev || host_ptr ? count : static_cast<int64_t>(host.size()); }
  };
  Index x_index, index;
};

struct Hints {
  // Only these source ranges (elements, relative to the source base) hold data;
  // the rest need not be copied. Empty means everything is valid.
  std::vector<i64> valid_start, valid_len;
  bool has_valid = false;
  // Bucketing context for the tuner, e.g. "tp=2".
  std::string ctx;
  // The frontend's own implementation of this copy (see NativeOp); null when
  // the frontend has no equivalent with identical results.
  std::shared_ptr<NativeOp> native;
  // The caller does not need the host side complete on return (an
  // intercepted copy_ with non_blocking=True); device order still holds.
  bool host_async = false;
  // The caller checks, before each run, that the copy's source and
  // destination do not share bytes (a step plan with host indices,
  // Frozen::host_overlap): footprints that may overlap are not refused.
  bool overlap_checked = false;
};

}  // namespace ce
