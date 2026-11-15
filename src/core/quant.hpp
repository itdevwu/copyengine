#pragma once

// Unit functions of the tile IR's quantization catalog (docs/design/v0.4.md),
// shared verbatim by the host backend and the CUDA kernels. One call handles
// one unit: it reads only that unit's inputs (the admission line's rule 2).
// Every function reproduces the torch recipe of its format bit for bit
// (tests/python/test_quant_ref.py); float32 arithmetic, no FMA.

#include "core/convert.hpp"

namespace ce::cv {

enum class QFmt : int { fp8_e4m3 = 0, fp8_e5m2, int8, mxfp8_e4m3, mxfp8_e5m2, nvfp4, int4, uint4 };

// Packed formats store two values per byte, element 2i in the low nibble.
constexpr CE_HD bool qfmt_packed(QFmt f) { return f == QFmt::nvfp4 || f == QFmt::int4 || f == QFmt::uint4; }

CE_HD bool qfmt_mx(QFmt f) { return f == QFmt::mxfp8_e4m3 || f == QFmt::mxfp8_e5m2; }
CE_HD float qmax_of(QFmt f) {
  switch (f) {
    case QFmt::fp8_e4m3: case QFmt::mxfp8_e4m3: return 448.0f;
    case QFmt::fp8_e5m2: case QFmt::mxfp8_e5m2: return 57344.0f;
    case QFmt::int8: return 127.0f;
    case QFmt::int4: return 7.0f;
    case QFmt::uint4: return 15.0f;
    default: return 6.0f;  // nvfp4 (E2M1)
  }
}

CE_HD bool is_nan(float x) { return x != x; }

// IEEE float32 product and quotient, round to nearest: on the GPU through
// the _rn intrinsics, which the compiler may not reassociate (it otherwise
// turns a * (1 / b) into a / b, a different rounding).
CE_HD float mul_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fmul_rn(a, b);
#else
  return a * b;
#endif
}
CE_HD float div_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
  return __fdiv_rn(a, b);
#else
  return a / b;
#endif
}
CE_HD float abs_f(float x) { return u2f(f2u(x) & 0x7FFFFFFFu); }
// torch.amax: NaN wins.
CE_HD float amax_step(float m, float v) { return is_nan(m) ? m : is_nan(v) ? v : (v > m ? v : m); }
// torch.amin: NaN wins.
CE_HD float amin_step(float m, float v) { return is_nan(m) ? m : is_nan(v) ? v : (v < m ? v : m); }
// torch.clamp: NaN stays NaN.
CE_HD float clamp_f(float v, float lo, float hi) { return is_nan(v) ? v : v < lo ? lo : v > hi ? hi : v; }

// One element of a scaled format: `v` already divided by its scale.
CE_HD uint8_t quant_scaled(float v, QFmt f) {
  const float m = qmax_of(f);
  if (f == QFmt::int8) {  // torch.round (half to even) then clamp; NaN -> 0 (torch leaves it undefined)
    if (is_nan(v)) return 0;
    const float r = clamp_f(rintf(v), -m, m);
    return static_cast<uint8_t>(static_cast<int8_t>(r));
  }
  const float c = clamp_f(v, -m, m);
  return (f == QFmt::fp8_e4m3 || f == QFmt::mxfp8_e4m3) ? float_to_f8e4m3(c) : float_to_f8e5m2(c);
}

CE_HD float dequant_scaled(uint8_t q, QFmt f) {
  if (f == QFmt::int8) return static_cast<float>(static_cast<int8_t>(q));
  return (f == QFmt::fp8_e4m3 || f == QFmt::mxfp8_e4m3) ? f8e4m3_to_float(q) : f8e5m2_to_float(q);
}

// ---- E2M1 (NVFP4 elements) ----------------------------------------------------------

// Nearest even on the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}; NaN -> +0.
CE_HD uint8_t float_to_e2m1(float v) {
  if (is_nan(v)) return 0;
  const uint8_t s = (f2u(v) >> 28) & 0x8u;
  const float m = abs_f(v);
  uint8_t c;
  if (m <= 0.25f) c = 0;
  else if (m < 0.75f) c = 1;
  else if (m <= 1.25f) c = 2;
  else if (m < 1.75f) c = 3;
  else if (m <= 2.5f) c = 4;
  else if (m < 3.5f) c = 5;
  else if (m <= 5.0f) c = 6;
  else c = 7;
  return static_cast<uint8_t>(s | c);
}

CE_HD float e2m1_to_float(uint8_t c) {
  const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float m = mag[c & 7u];
  return (c & 8u) ? -m : m;
}

// ---- units ----------------------------------------------------------------------

// What quantizing one unit needs once its reductions are known: the divisor
// of each value, the scale to store (float32: scaled formats and INT4; a byte:
// E8M0 for MX, E4M3 for NVFP4, the zero point for uint4), and whether every
// value is zero (NVFP4 with a zero scale). Computed from the unit's amax, and
// for uint4 its minimum and maximum (both clamped to include 0).
struct UnitQ {
  float div = 1.0f;
  float scale = 0.0f;
  uint8_t bits = 0;
  bool zero_all = false;
};

CE_HD bool qfmt_needs_range(QFmt f) { return f == QFmt::uint4; }

CE_HD UnitQ unit_params(QFmt f, float amax, float lo, float hi, float eps, float gs) {
  UnitQ u;
  auto floor_eps = [&](float v) { return is_nan(v) ? v : (v < eps ? eps : v); };
  switch (f) {
    // Constants divide as products with their float32 reciprocal: what torch
    // computes for tensor / scalar on the GPU, and the same on every device.
    case QFmt::fp8_e4m3: case QFmt::fp8_e5m2: case QFmt::int8: case QFmt::int4:
      u.scale = u.div = mul_rn(floor_eps(amax), div_rn(1.0f, qmax_of(f)));
      return u;
    case QFmt::uint4: {
      u.scale = u.div = mul_rn(floor_eps(hi - lo), div_rn(1.0f, 15.0f));
      const float zf = clamp_f(rintf(div_rn(-lo, u.div)), 0.0f, 15.0f);
      u.bits = is_nan(zf) ? 0 : static_cast<uint8_t>(zf);
      return u;
    }
    case QFmt::mxfp8_e4m3: case QFmt::mxfp8_e5m2: {
      const int emax = f == QFmt::mxfp8_e4m3 ? 8 : 15;
      const uint32_t b = f2u(amax);
      const uint32_t e = (b >> 23) & 0xFFu, man = b & 0x7FFFFFu;
      if (e == 0xFFu) {  // NaN or Inf: no scale (0xFF, NaN) and NaN values
        u.bits = 0xFF;
        u.div = u2f(0x7FC00000u);
        return u;
      }
      int shared;
      if (b == 0) shared = -127;
      else {
        const int fl = e ? static_cast<int>(e) - 127 : 31 - static_cast<int>(clz32(man)) - 149;  // floor(log2)
        shared = fl - emax;
        shared = shared < -127 ? -127 : shared > 127 ? 127 : shared;
      }
      u.bits = static_cast<uint8_t>(shared + 127);
      u.div = shared >= -126 ? u2f(static_cast<uint32_t>(shared + 127) << 23) : u2f(0x00400000u);
      return u;
    }
    case QFmt::nvfp4: {
      float sv = mul_rn(mul_rn(amax, div_rn(1.0f, 6.0f)), div_rn(1.0f, gs));  // .clamp(max=448)
      if (!is_nan(sv) && sv > 448.0f) sv = 448.0f;
      u.bits = float_to_f8e4m3(sv);
      const float sf = f8e4m3_to_float(u.bits);
      u.zero_all = sf == 0.0f;
      u.div = mul_rn(sf, gs);
      return u;
    }
  }
  return u;
}

// One value's code: a byte, or a nibble for the packed formats.
CE_HD uint8_t quant_elem(QFmt f, float x, const UnitQ& u) {
  switch (f) {
    case QFmt::int4: {
      const float v = div_rn(x, u.div);
      return is_nan(v) ? 0 : static_cast<uint8_t>(static_cast<int>(clamp_f(rintf(v), -8.0f, 7.0f)) & 0xF);
    }
    case QFmt::uint4: {
      const float v = rintf(div_rn(x, u.div)) + static_cast<float>(u.bits);
      return is_nan(v) ? 0 : static_cast<uint8_t>(clamp_f(v, 0.0f, 15.0f));
    }
    case QFmt::nvfp4:
      return u.zero_all ? 0 : float_to_e2m1(clamp_f(div_rn(x, u.div), -6.0f, 6.0f));
    default:
      return quant_scaled(div_rn(x, u.div), f);
  }
}

// Dynamic quantization of one unit of `g` values (strided by `xs`) into `q`
// (strided by `qs`; packed formats: bytes of two values) and its scale
// (*scale: float32; *scale_bits: E8M0 / E4M3 byte or uint4's zero point).
// The host path, unit by unit; kernels share unit_params and quant_elem.
template <class In>
CE_HD void quantize_unit(In x, long long xs, int g, QFmt f, float eps, float gs, uint8_t* q, long long qs,
                         float* scale, uint8_t* scale_bits) {
  float amax = 0.0f, lo = 0.0f, hi = 0.0f;
  for (int i = 0; i < g; ++i) {
    const float v = x(i * xs);
    amax = amax_step(amax, abs_f(v));
    if (qfmt_needs_range(f)) lo = amin_step(lo, v), hi = amax_step(hi, v);
  }
  const UnitQ u = unit_params(f, amax, lo, hi, eps, gs);
  *scale = u.scale;
  *scale_bits = u.bits;
  if (qfmt_packed(f)) {
    for (int i = 0; i + 1 < g; i += 2)
      q[(i / 2) * qs] =
          static_cast<uint8_t>(quant_elem(f, x(i * xs), u) | (quant_elem(f, x((i + 1) * xs), u) << 4));
    return;
  }
  for (int i = 0; i < g; ++i) q[i * qs] = quant_elem(f, x(i * xs), u);
}

// The scale of one unit as a float32 factor (dequantization multiplies by it).
CE_HD float unit_scale(QFmt f, float scale, uint8_t scale_bits, float gs) {
  if (qfmt_mx(f)) return f8e8m0_to_float(scale_bits);
  if (f == QFmt::nvfp4) return mul_rn(f8e4m3_to_float(scale_bits), gs);
  return scale;
}

// Element i of a dequantized unit (`q` as stored: packed formats two per
// byte); `zero`: uint4's zero point (the unit's scale byte).
CE_HD float dequantize_at(const uint8_t* q, long long qs, int i, QFmt f, float factor, uint8_t zero = 0) {
  if (qfmt_packed(f)) {
    const uint8_t b = q[(i / 2) * qs];
    const uint8_t n = (i & 1) ? static_cast<uint8_t>(b >> 4) : static_cast<uint8_t>(b & 0xFu);
    if (f == QFmt::nvfp4) return mul_rn(e2m1_to_float(n), factor);
    if (f == QFmt::int4) return mul_rn(static_cast<float>(static_cast<int>(n) - ((n & 8u) ? 16 : 0)), factor);
    return mul_rn(static_cast<float>(n) - static_cast<float>(zero), factor);  // uint4
  }
  return mul_rn(dequant_scaled(q[i * qs], f), factor);
}

}  // namespace ce::cv
