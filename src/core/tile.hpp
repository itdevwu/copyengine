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
  int static_scale = 0;  // quantize with the given scale (and zero point): they are read, not written
  // Kernels may move 4 consecutive values at a time: x and q contiguous along
  // the unit, every unit's x (8 / 16 bytes) and q (4 / 2 bytes) aligned, and
  // g a multiple of 4 (set when queued).
  int vec = 0;
  // Units within one row of the first dimension are back to back in every
  // tensor (x, q, s, z): unit u of that row is at a constant step, so finding
  // it takes one division (set when queued; indices still select the row).
  int flat = 0;
  int bh = 1;  // box units: bh rows of g values, xp / qp bytes apart (byte formats with float scales)
  long long xp = 0, qp = 0;
  float eps = 1e-12f, gs = 1.0f;
  // Gathers / scatters: the first row dimension of x (xi), or of the
  // quantized side q, s, z (ci), read through an int64 index table (0: none),
  // each index below its bound (xb, cb). A unit with an index out of range is
  // skipped and sets the word at `err` (device tables; host ones are checked
  // when queued).
  long long xi = 0, ci = 0, xb = 0, cb = 0, err = 0;

  CE_HD long long units() const { return rows[0] * rows[1] * rows[2] * upr; }
};

CE_HD void* addr(long long a) { return reinterpret_cast<void*>(static_cast<uintptr_t>(a)); }

// Byte offsets of unit u's row in each tensor, and its index in the row;
// `bad`: an index table held an index out of range (the unit is skipped).
struct UnitAt {
  long long x, q, s, z;
  long long j;
  bool bad;
};

CE_HD UnitAt unit_at(const TileK& t, long long u) {
  UnitAt a;
  a.bad = false;
  if (t.flat) {  // one division: the row of the first dimension, then constant steps
    const long long per = t.rows[1] * t.rows[2] * t.upr;
    long long i0, r;
    if (u < 0x7FFFFFFFLL && per < 0x7FFFFFFFLL) {
      const uint32_t uu = static_cast<uint32_t>(u), pp = static_cast<uint32_t>(per);
      i0 = uu / pp, r = uu - static_cast<uint32_t>(i0) * pp;
    } else {
      i0 = u / per, r = u - i0 * per;
    }
    long long ix = i0, ic = i0;
    if (t.xi) {
      ix = rd<long long>(addr(t.xi + i0 * 8));
      a.bad = a.bad || ix < 0 || ix >= t.xb;
    }
    if (t.ci) {
      ic = rd<long long>(addr(t.ci + i0 * 8));
      a.bad = a.bad || ic < 0 || ic >= t.cb;
    }
    const long long vals = static_cast<long long>(t.g);
    const long long qstep = qfmt_packed(static_cast<QFmt>(t.fmt)) ? vals / 2 : vals;
    a.j = 0;
    a.x = t.x + ix * t.xr[0] + r * vals * t.xe;
    a.q = t.q + ic * t.qr[0] + r * qstep * t.qe;
    a.s = t.s + ic * t.sr[0] + r * t.se;
    a.z = t.z + ic * t.zr[0] + r * t.ze;
    return a;
  }
  long long i0, i1, i2;
  if (u < 0x7FFFFFFFLL && t.rows[1] * t.rows[2] < 0x7FFFFFFFLL) {  // 32-bit divisions: a few times cheaper
    const uint32_t uu = static_cast<uint32_t>(u), up = static_cast<uint32_t>(t.upr);
    const uint32_t r2 = static_cast<uint32_t>(t.rows[2]), r1 = static_cast<uint32_t>(t.rows[1]);
    uint32_t r = uu / up;
    a.j = uu - r * up;
    const uint32_t q2 = r / r2;
    i2 = r - q2 * r2;
    const uint32_t q1 = q2 / r1;
    i1 = q2 - q1 * r1, i0 = q1;
  } else {
    a.j = u % t.upr;
    long long r = u / t.upr;
    i2 = r % t.rows[2];
    r /= t.rows[2];
    i1 = r % t.rows[1], i0 = r / t.rows[1];
  }
  long long ix = i0, ic = i0;
  if (t.xi) {
    ix = rd<long long>(addr(t.xi + i0 * 8));
    a.bad = a.bad || ix < 0 || ix >= t.xb;
  }
  if (t.ci) {
    ic = rd<long long>(addr(t.ci + i0 * 8));
    a.bad = a.bad || ic < 0 || ic >= t.cb;
  }
  a.x = t.x + ix * t.xr[0] + i1 * t.xr[1] + i2 * t.xr[2];
  a.q = t.q + ic * t.qr[0] + i1 * t.qr[1] + i2 * t.qr[2];
  a.s = t.s + ic * t.sr[0] + i1 * t.sr[1] + i2 * t.sr[2] + a.j * t.se;
  a.z = t.z + ic * t.zr[0] + i1 * t.zr[1] + i2 * t.zr[2] + a.j * t.ze;
  return a;
}

// Value k of unit `a` in x, and the code byte holding it in q.
CE_HD long long x_at(const TileK& t, const UnitAt& a, int k) { return a.x + (a.j * t.g + k) * t.xe; }
CE_HD long long q_at(const TileK& t, const UnitAt& a, int k) {
  const QFmt f = static_cast<QFmt>(t.fmt);
  return qfmt_packed(f) ? a.q + ((a.j * t.g + k) / 2) * t.qe : a.q + (a.j * t.g + k) * t.qe;
}

// Value (r, c) of a box unit in x, and its code byte in q.
CE_HD long long xb_at(const TileK& t, const UnitAt& a, int r, int c) { return a.x + r * t.xp + (a.j * t.g + c) * t.xe; }
CE_HD long long qb_at(const TileK& t, const UnitAt& a, int r, int c) { return a.q + r * t.qp + (a.j * t.g + c) * t.qe; }

// A box unit (byte formats, float32 scale) on the host: the same unit
// functions over its bh x g values.
CE_HD void tile_box_unit(const TileK& t, const UnitAt& a) {
  const QFmt f = static_cast<QFmt>(t.fmt);
  auto xv = [&](int r, int c) { return val_float(load(addr(xb_at(t, a, r, c)), t.xdt)); };
  if (!t.quantize) {
    const float factor = unit_scale(f, rd<float>(addr(a.s)), 0, t.gs);
    for (int r = 0; r < t.bh; ++r)
      for (int c = 0; c < t.g; ++c)
        store_float(addr(xb_at(t, a, r, c)), t.xdt, dequantize_code(f, rd<uint8_t>(addr(qb_at(t, a, r, c))), factor));
    return;
  }
  UnitQ p;
  if (t.static_scale) {
    p = unit_from_scale(f, rd<float>(addr(a.s)), 0, 0, t.gs);
  } else {
    float amax = 0.0f;
    for (int r = 0; r < t.bh; ++r)
      for (int c = 0; c < t.g; ++c) amax = amax_step(amax, abs_f(xv(r, c)));
    p = unit_params(f, amax, 0.0f, 0.0f, t.eps, t.gs);
    wr<float>(addr(a.s), p.scale);
  }
  for (int r = 0; r < t.bh; ++r)
    for (int c = 0; c < t.g; ++c) wr<uint8_t>(addr(qb_at(t, a, r, c)), quant_elem(f, xv(r, c), p));
}

// The host path: one unit, sequentially (kernels split it over a warp).
CE_HD void tile_unit(const TileK& t, long long u) {
  const QFmt f = static_cast<QFmt>(t.fmt);
  const UnitAt a = unit_at(t, u);
  if (a.bad) return;  // host tables are checked when queued
  if (t.bh > 1) {
    tile_box_unit(t, a);
    return;
  }
  if (t.quantize && t.static_scale) {
    const float s = t.scale_byte ? 0.0f : rd<float>(addr(a.s));
    const uint8_t sb = t.scale_byte ? rd<uint8_t>(addr(a.s)) : 0;
    const uint8_t z = f == QFmt::uint4 ? rd<uint8_t>(addr(a.z)) : 0;
    const UnitQ p = unit_from_scale(f, s, sb, z, t.gs);
    for (int k = 0; k < t.g; ++k) {
      const uint8_t c = quant_elem(f, val_float(load(addr(x_at(t, a, k)), t.xdt)), p);
      uint8_t* qb = static_cast<uint8_t*>(addr(q_at(t, a, k)));
      if (qfmt_packed(f)) *qb = static_cast<uint8_t>((k & 1) ? ((*qb & 0x0Fu) | (c << 4)) : ((*qb & 0xF0u) | c));
      else *qb = c;
    }
    return;
  }
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
