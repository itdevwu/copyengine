#pragma once

// Element conversion shared verbatim by the host backend and the CUDA kernels,
// so both produce identical bits. Pure integer / IEEE arithmetic only: no
// intrinsics, no FMA (callers compile with -ffp-contract=off / --fmad=false).

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <type_traits>

#if defined(__CUDACC__)
#define CE_HD __host__ __device__ __forceinline__
#else
#define CE_HD inline
#endif

namespace ce::cv {

struct half_t { uint16_t bits; };
struct bf16_t { uint16_t bits; };
struct bool_t { uint8_t v; };

CE_HD uint32_t f2u(float f) {
#if defined(__CUDA_ARCH__)
  return __float_as_uint(f);
#else
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
#endif
}

CE_HD float u2f(uint32_t u) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(u);
#else
  float f;
  memcpy(&f, &u, 4);
  return f;
#endif
}

// IEEE binary16 <-> binary32, round to nearest even (after Maratyszcza/FP16).
CE_HD float half_to_float(uint16_t h) {
  const uint32_t w = static_cast<uint32_t>(h) << 16;
  const uint32_t sign = w & 0x80000000u;
  const uint32_t two_w = w + w;
  const float normalized = u2f((two_w >> 4) + (0xE0u << 23)) * u2f(0x07800000u);  // * 2^-112
  const float denormalized = u2f((two_w >> 17) | (126u << 23)) - 0.5f;
  const uint32_t r = sign | (two_w < (1u << 27) ? f2u(denormalized) : f2u(normalized));
  return u2f(r);
}

CE_HD uint16_t float_to_half(float f) {
  const float to_inf = u2f(0x77800000u);   // 2^112
  const float to_zero = u2f(0x08800000u);  // 2^-110
  const uint32_t w = f2u(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & 0x80000000u;
  float base = (u2f(w & 0x7FFFFFFFu) * to_inf) * to_zero;
  uint32_t bias = shl1_w & 0xFF000000u;
  if (bias < 0x71000000u) bias = 0x71000000u;
  base = u2f((bias >> 1) + 0x07800000u) + base;
  const uint32_t b = f2u(base);
  const uint32_t nonsign = ((b >> 13) & 0x00007C00u) + (b & 0x00000FFFu);
  return static_cast<uint16_t>((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

CE_HD float bf16_to_float(uint16_t h) { return u2f(static_cast<uint32_t>(h) << 16); }

CE_HD uint16_t float_to_bf16(float f) {
  const uint32_t u = f2u(f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return static_cast<uint16_t>((u >> 16) | 0x0040u);  // quiet NaN
  return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// ---- type traits ----------------------------------------------------------------

template <class T> struct is_int { static constexpr bool value = false; };
template <> struct is_int<int64_t> { static constexpr bool value = true; };
template <> struct is_int<int32_t> { static constexpr bool value = true; };
template <> struct is_int<int16_t> { static constexpr bool value = true; };
template <> struct is_int<int8_t> { static constexpr bool value = true; };
template <> struct is_int<uint8_t> { static constexpr bool value = true; };

template <class T> struct limits;
template <> struct limits<int64_t> { static constexpr int64_t lo = -9223372036854775807LL - 1, hi = 9223372036854775807LL; };
template <> struct limits<int32_t> { static constexpr int32_t lo = -2147483647 - 1, hi = 2147483647; };
template <> struct limits<int16_t> { static constexpr int16_t lo = -32768, hi = 32767; };
template <> struct limits<int8_t> { static constexpr int8_t lo = -128, hi = 127; };
template <> struct limits<uint8_t> { static constexpr uint8_t lo = 0, hi = 255; };

// Widen a float-like value: halves to float, float/double unchanged.
CE_HD float widen(half_t x) { return half_to_float(x.bits); }
CE_HD float widen(bf16_t x) { return bf16_to_float(x.bits); }
CE_HD float widen(float x) { return x; }
CE_HD double widen(double x) { return x; }

// Saturating float -> int, NaN -> 0, truncation toward zero.
template <class I, class F>
CE_HD I sat(F x) {
  if (!(x == x)) return 0;
  if (x <= static_cast<F>(limits<I>::lo)) return limits<I>::lo;
  if (x >= static_cast<F>(limits<I>::hi)) return limits<I>::hi;
  return static_cast<I>(x);
}

template <class T> CE_HD T from_float(float v);
template <> CE_HD double from_float<double>(float v) { return v; }
template <> CE_HD float from_float<float>(float v) { return v; }
template <> CE_HD half_t from_float<half_t>(float v) { return {float_to_half(v)}; }
template <> CE_HD bf16_t from_float<bf16_t>(float v) { return {float_to_bf16(v)}; }
template <> CE_HD bool_t from_float<bool_t>(float v) { return {static_cast<uint8_t>(v != 0.0f)}; }
template <> CE_HD int64_t from_float<int64_t>(float v) { return sat<int64_t>(v); }
template <> CE_HD int32_t from_float<int32_t>(float v) { return sat<int32_t>(v); }
template <> CE_HD int16_t from_float<int16_t>(float v) { return sat<int16_t>(v); }
template <> CE_HD int8_t from_float<int8_t>(float v) { return sat<int8_t>(v); }
template <> CE_HD uint8_t from_float<uint8_t>(float v) { return sat<uint8_t>(v); }

template <class T> CE_HD float to_float(T x) {
  if constexpr (is_int<T>::value) return static_cast<float>(x);
  else return static_cast<float>(widen(x));
}
template <> CE_HD float to_float<bool_t>(bool_t x) { return x.v ? 1.0f : 0.0f; }

template <class T> struct is_halfish { static constexpr bool value = false; };
template <> struct is_halfish<half_t> { static constexpr bool value = true; };
template <> struct is_halfish<bf16_t> { static constexpr bool value = true; };

// Identity-function conversion S -> D (see the rules in copyengine/types.hpp).
template <class D, class S>
CE_HD D cast(S s) {
  if constexpr (sizeof(D) == sizeof(S) && std::is_same_v<D, S>) {
    return s;
  } else if constexpr (std::is_same_v<D, bool_t>) {
    if constexpr (std::is_same_v<S, bool_t>) return s;
    else if constexpr (is_int<S>::value) return {static_cast<uint8_t>(s != 0)};
    else return {static_cast<uint8_t>(widen(s) != 0)};
  } else if constexpr (std::is_same_v<S, bool_t>) {
    if constexpr (is_int<D>::value) return static_cast<D>(s.v ? 1 : 0);
    else return from_float<D>(s.v ? 1.0f : 0.0f);
  } else if constexpr (is_int<S>::value) {
    if constexpr (is_int<D>::value) return static_cast<D>(s);
    else if constexpr (is_halfish<D>::value) return from_float<D>(static_cast<float>(s));
    else return static_cast<D>(s);
  } else {
    // S is float-like
    if constexpr (is_int<D>::value) return sat<D>(widen(s));
    else if constexpr (std::is_same_v<D, double>) return static_cast<double>(widen(s));
    else if constexpr (std::is_same_v<D, float>) return static_cast<float>(widen(s));
    else return from_float<D>(static_cast<float>(widen(s)));
  }
}

struct Fn {
  float scale, bias, lo, hi;
  bool round, clamp;
};

// Non-identity function: computed in f32, no FMA.
template <class D, class S>
CE_HD D apply(S s, const Fn& f) {
  float v = to_float(s);
  v = v * f.scale;
  v = v + f.bias;
  if (f.round) v = rintf(v);
  if (f.clamp) v = fminf(fmaxf(v, f.lo), f.hi);
  return from_float<D>(v);
}

}  // namespace ce::cv
