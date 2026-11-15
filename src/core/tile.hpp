#pragma once

// Tile ops (docs/design/v0.4.md): quantization and dequantization while
// copying, unit by unit. TileK describes one op with absolute byte addresses;
// tile_unit runs one unit. Shared verbatim by the host backend and the CUDA
// kernels (which run the same unit functions one warp per unit).

#include "core/quant.hpp"

namespace ce::cv {

// Rows (up to three leading dimensions) of `upr` units of `g` values along
// the innermost dimension. x: the full-precision tensor (read when
// quantizing, written when dequantizing); q: the codes (packed formats: one
// byte per two values, `qe` the step between bytes); s: one scale per unit
// (float32, or a byte: E8M0 / E4M3); z: uint4's zero point per unit.
struct TileK {
  long long x = 0, q = 0, s = 0, z = 0;
  long long rows[3] = {1, 1, 1};
  long long xr[3] = {0, 0, 0}, qr[3] = {0, 0, 0}, sr[3] = {0, 0, 0}, zr[3] = {0, 0, 0};
  long long xe = 0, qe = 0, se = 0, ze = 0;
  int g = 1, upr = 1;
  int fmt = 0;         // QFmt
  int xdt = 1;         // x's dtype (convert.hpp codes: f32, f16, bf16)
  int scale_byte = 0;  // the scale is stored as a byte (MX, NVFP4)
  int quantize = 1;    // else dequantize
  float eps = 1e-12f, gs = 1.0f;

  CE_HD long long units() const { return rows[0] * rows[1] * rows[2] * upr; }
};

// Byte offsets of unit u's row in each tensor, and its index in the row.
struct UnitAt {
  long long x, q, s, z;
  long long j;
};

CE_HD UnitAt unit_at(const TileK& t, long long u) {
  UnitAt a;
  a.j = u % t.upr;
  long long r = u / t.upr;
  const long long i2 = r % t.rows[2];
  r /= t.rows[2];
  const long long i1 = r % t.rows[1], i0 = r / t.rows[1];
  a.x = t.x + i0 * t.xr[0] + i1 * t.xr[1] + i2 * t.xr[2];
  a.q = t.q + i0 * t.qr[0] + i1 * t.qr[1] + i2 * t.qr[2];
  a.s = t.s + i0 * t.sr[0] + i1 * t.sr[1] + i2 * t.sr[2] + a.j * t.se;
  a.z = t.z + i0 * t.zr[0] + i1 * t.zr[1] + i2 * t.zr[2] + a.j * t.ze;
  return a;
}

CE_HD void* addr(long long a) { return reinterpret_cast<void*>(static_cast<uintptr_t>(a)); }

// Value k of unit `a` in x, and the code byte holding it in q.
CE_HD long long x_at(const TileK& t, const UnitAt& a, int k) { return a.x + (a.j * t.g + k) * t.xe; }
CE_HD long long q_at(const TileK& t, const UnitAt& a, int k) {
  const QFmt f = static_cast<QFmt>(t.fmt);
  return qfmt_packed(f) ? a.q + ((a.j * t.g + k) / 2) * t.qe : a.q + (a.j * t.g + k) * t.qe;
}

// The host path: one unit, sequentially (kernels split it over a warp).
CE_HD void tile_unit(const TileK& t, long long u) {
  const QFmt f = static_cast<QFmt>(t.fmt);
  const UnitAt a = unit_at(t, u);
  if (t.quantize) {
    float s = 0.0f;
    uint8_t bits = 0;
    uint8_t* q0 = static_cast<uint8_t*>(addr(q_at(t, a, 0)));
    quantize_unit([&](long long k) { return val_float(load(addr(x_at(t, a, static_cast<int>(k))), t.xdt)); }, 1, t.g,
                  f, t.eps, t.gs, q0, t.qe, &s, &bits);
    if (t.scale_byte) wr<uint8_t>(addr(a.s), bits);
    else wr<float>(addr(a.s), s);
    if (f == QFmt::uint4) wr<uint8_t>(addr(a.z), bits);
    return;
  }
  const float s = t.scale_byte ? 0.0f : rd<float>(addr(a.s));
  const uint8_t bits = t.scale_byte ? rd<uint8_t>(addr(a.s)) : (f == QFmt::uint4 ? rd<uint8_t>(addr(a.z)) : 0);
  const float factor = unit_scale(f, s, bits, t.gs);
  const uint8_t* q0 = static_cast<const uint8_t*>(addr(q_at(t, a, 0)));
  for (int k = 0; k < t.g; ++k)
    store_float(addr(x_at(t, a, k)), t.xdt, dequantize_at(q0, t.qe, k, f, factor, bits));
}

}  // namespace ce::cv
