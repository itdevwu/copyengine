#pragma once

// Unit functions of the tile IR's quantization catalog (docs/design/v0.4.md),
// shared verbatim by the host backend and the CUDA kernels. One call handles
// one unit: it reads only that unit's inputs (the admission line's rule 2).
// Every function reproduces the torch recipe of its format bit for bit
// (tests/python/test_quant_ref.py); float32 arithmetic, no FMA.

#include "core/convert.hpp"

namespace ce::cv {

enum class QFmt : int { fp8_e4m3 = 0, fp8_e5m2, int8, mxfp8_e4m3, mxfp8_e5m2, nvfp4 };

CE_HD bool qfmt_mx(QFmt f) { return f == QFmt::mxfp8_e4m3 || f == QFmt::mxfp8_e5m2; }
CE_HD float qmax_of(QFmt f) {
  switch (f) {
    case QFmt::fp8_e4m3: case QFmt::mxfp8_e4m3: return 448.0f;
    case QFmt::fp8_e5m2: case QFmt::mxfp8_e5m2: return 57344.0f;
    case QFmt::int8: return 127.0f;
    default: return 6.0f;  // nvfp4 (E2M1)
  }
}

CE_HD bool is_nan(float x) { return x != x; }
CE_HD float abs_f(float x) { return u2f(f2u(x) & 0x7FFFFFFFu); }
// torch.amax: NaN wins.
CE_HD float amax_step(float m, float v) { return is_nan(m) ? m : is_nan(v) ? v : (v > m ? v : m); }
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

// Dynamic quantization of one unit of `g` values (strided by `xs`) into `q`
// (strided by `qs`; nvfp4: packed bytes, two values each) and its scale:
// float32 for scaled formats, the E8M0 / E4M3 byte for MX / NVFP4 (in
// *scale_bits). `gs`: NVFP4's global scale. `eps`: the scaled formats' floor.
template <class In>
CE_HD void quantize_unit(In x, long long xs, int g, QFmt f, float eps, float gs, uint8_t* q, long long qs,
                         float* scale, uint8_t* scale_bits) {
  float amax = 0.0f;
  for (int i = 0; i < g; ++i) amax = amax_step(amax, abs_f(x(i * xs)));
  if (f == QFmt::fp8_e4m3 || f == QFmt::fp8_e5m2 || f == QFmt::int8) {
    const float s = (is_nan(amax) ? amax : (amax < eps ? eps : amax)) / qmax_of(f);
    *scale = s;
    for (int i = 0; i < g; ++i) q[i * qs] = quant_scaled(x(i * xs) / s, f);
    return;
  }
  if (qfmt_mx(f)) {
    const int emax = f == QFmt::mxfp8_e4m3 ? 8 : 15;
    const uint32_t b = f2u(amax);
    const uint32_t e = (b >> 23) & 0xFFu, man = b & 0x7FFFFFu;
    float sf;
    if (e == 0xFFu) {  // NaN or Inf: no scale (0xFF, NaN) and NaN values
      *scale_bits = 0xFF;
      sf = u2f(0x7FC00000u);
    } else {
      int shared;
      if (b == 0) shared = -127;
      else {
        const int fl = e ? static_cast<int>(e) - 127 : 31 - static_cast<int>(clz32(man)) - 149;  // floor(log2)
        shared = fl - emax;
        shared = shared < -127 ? -127 : shared > 127 ? 127 : shared;
      }
      *scale_bits = static_cast<uint8_t>(shared + 127);
      sf = shared >= -126 ? u2f(static_cast<uint32_t>(shared + 127) << 23) : u2f(0x00400000u);
    }
    for (int i = 0; i < g; ++i) q[i * qs] = quant_scaled(x(i * xs) / sf, f);
    return;
  }
  // nvfp4
  float sv = amax / 6.0f / gs;  // .clamp(max=448)
  if (!is_nan(sv) && sv > 448.0f) sv = 448.0f;
  const uint8_t sb = float_to_f8e4m3(sv);
  *scale_bits = sb;
  const float sf = f8e4m3_to_float(sb);
  const float den = sf * gs;
  for (int i = 0; i + 1 < g; i += 2) {
    const uint8_t lo = sf == 0.0f ? 0 : float_to_e2m1(clamp_f(x(i * xs) / den, -6.0f, 6.0f));
    const uint8_t hi = sf == 0.0f ? 0 : float_to_e2m1(clamp_f(x((i + 1) * xs) / den, -6.0f, 6.0f));
    q[(i / 2) * qs] = static_cast<uint8_t>(lo | (hi << 4));
  }
}

// The scale of one unit as a float32 factor (dequantization multiplies by it).
CE_HD float unit_scale(QFmt f, float scale, uint8_t scale_bits, float gs) {
  if (qfmt_mx(f)) return f8e8m0_to_float(scale_bits);
  if (f == QFmt::nvfp4) return f8e4m3_to_float(scale_bits) * gs;
  return scale;
}

// Element i of a dequantized unit (`q` as stored: nvfp4 packed).
CE_HD float dequantize_at(const uint8_t* q, long long qs, int i, QFmt f, float factor) {
  if (f == QFmt::nvfp4) {
    const uint8_t b = q[(i / 2) * qs];
    return e2m1_to_float((i & 1) ? (b >> 4) : (b & 0xFu)) * factor;
  }
  return dequant_scaled(q[i * qs], f) * factor;
}

}  // namespace ce::cv
