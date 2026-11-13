#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "copyengine/ir.hpp"

namespace ce {

enum class DType : uint8_t { f64, f32, f16, bf16, i64, i32, i16, i8, u8, b8 };
inline constexpr int kNumDTypes = 10;

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
struct Endpoint {
  Device device;
  uintptr_t base = 0;
  int64_t nbytes = 0;
  DType dtype = DType::u8;
  bool pinned = false;  // host only: page-locked
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

struct Hints {
  // Only these source ranges (elements, relative to the source base) hold data;
  // the rest need not be copied. Empty means everything is valid.
  std::vector<i64> valid_start, valid_len;
  bool has_valid = false;
  // Bucketing context for the tuner, e.g. "tp=2".
  std::string ctx;
  // Frontend handle that runs this copy natively (see NativeRunner); null
  // when the frontend has no equivalent with identical results.
  std::shared_ptr<void> native;
};

}  // namespace ce
