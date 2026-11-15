"""The quantization catalog's torch recipes (docs/design/v0.4.md).

Each format's recipe is written here once, in torch ops: it defines the bits
copyengine produces (tests/python/test_quant_ref.py checks the host and GPU
paths against it), and it is the torch baseline a quantizing copy is
measured against.
"""

from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None

FMT = {"fp8_e4m3": 0, "fp8_e5m2": 1, "int8": 2, "mxfp8_e4m3": 3, "mxfp8_e5m2": 4, "nvfp4": 5, "int4": 6, "uint4": 7}
QMAX = {"fp8_e4m3": 448.0, "fp8_e5m2": 57344.0, "int8": 127.0, "mxfp8_e4m3": 448.0, "mxfp8_e5m2": 57344.0,
        "int4": 7.0, "uint4": 15.0}
PACKED = ("nvfp4", "int4", "uint4")
UNIT = {"mxfp8_e4m3": 32, "mxfp8_e5m2": 32, "nvfp4": 16}  # fixed unit sizes
EPS = 1e-12


def _fp8(fmt):
    return torch.float8_e4m3fn if fmt in ("fp8_e4m3", "mxfp8_e4m3") else torch.float8_e5m2


def _pack(n):
    return n[:, 0::2] | (n[:, 1::2] << 4)


def e2m1(v):
    m = v.abs()
    code = sum((m > 0.25, m >= 0.75, m > 1.25, m >= 1.75, m > 2.5, m >= 3.5, m > 5.0), torch.zeros_like(m)).to(
        torch.uint8)
    code = code | (torch.signbit(v).to(torch.uint8) << 3)
    return torch.where(v.isnan(), 0, code).to(torch.uint8)


def e2m1_value(code):
    mag = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=code.device)[(code & 7).long()]
    return torch.where((code & 8) != 0, -mag, mag)


# Divisions by constants are written as products with their reciprocal
# (rounded to float32 by torch): torch computes tensor / scalar that way on
# the GPU but divides on the CPU, so `amax / 448` would differ between devices.

def quantize(x, g, fmt, eps=EPS, gs=1.0):
    """Codes (uint8; packed formats two per byte), float32 scales or scale
    bytes, and uint4's zero points, per unit of `g` values of `x`."""
    xf = x.float().reshape(-1, g)
    amax = xf.abs().amax(-1, keepdim=True)
    scale = bits = None
    if fmt in ("fp8_e4m3", "fp8_e5m2", "int8"):
        scale = amax.clamp(min=eps) * (1 / QMAX[fmt])
        v = xf / scale
        if fmt == "int8":
            q = torch.where(v.isnan(), 0.0, torch.round(v).clamp(-127, 127)).to(torch.int8).view(torch.uint8)
        else:
            q = v.clamp(-QMAX[fmt], QMAX[fmt]).to(_fp8(fmt)).view(torch.uint8)
    elif fmt == "int4":
        scale = amax.clamp(min=eps) * (1 / 7)
        v = xf / scale
        n = torch.where(v.isnan(), 0.0, torch.round(v).clamp(-8, 7)).to(torch.int32) & 0xF
        q = _pack(n.to(torch.uint8))
    elif fmt == "uint4":
        lo = xf.amin(-1, keepdim=True).clamp(max=0)
        hi = xf.amax(-1, keepdim=True).clamp(min=0)
        scale = (hi - lo).clamp(min=eps) * (1 / 15)
        zf = torch.round(-lo / scale).clamp(0, 15)
        z = torch.where(zf.isnan(), 0.0, zf)
        v = torch.round(xf / scale) + z
        q = _pack(torch.where(v.isnan(), 0.0, v.clamp(0, 15)).to(torch.uint8))
        bits = z.to(torch.uint8)
    elif fmt in ("mxfp8_e4m3", "mxfp8_e5m2"):
        emax = 8 if fmt == "mxfp8_e4m3" else 15
        _, e = torch.frexp(amax)
        shared = (e - 1 - emax).clamp(-127, 127)
        shared = torch.where(amax == 0, -127, shared)
        finite = amax.isfinite()
        tiny = torch.tensor(0x00400000, dtype=torch.int32, device=x.device).view(torch.float32)
        sf = torch.where(shared >= -126, ((shared + 127).to(torch.int32) << 23).view(torch.float32), tiny)
        sf = torch.where(finite, sf, float("nan"))
        q = (xf / sf).clamp(-QMAX[fmt], QMAX[fmt]).to(_fp8(fmt)).view(torch.uint8)
        bits = torch.where(finite, shared + 127, 255).to(torch.uint8)
    elif fmt == "nvfp4":
        sv = (amax * (1 / 6) * (1 / gs)).clamp(max=448)
        s = sv.to(torch.float8_e4m3fn)
        sf = s.float()
        v = (xf / (sf * gs)).clamp(-6, 6)
        q = _pack(torch.where(sf == 0, 0, e2m1(v)).to(torch.uint8))
        bits = s.view(torch.uint8)
    else:
        raise ValueError(f"unknown format {fmt!r}")
    return q.reshape(-1), (None if scale is None else scale.reshape(-1)), (None if bits is None else bits.reshape(-1))


def dequantize(q, scale, bits, g, fmt, gs=1.0):
    """float32 values from codes (as `quantize` returns them) and the unit's
    scale (float32), scale bytes, or uint4's zero points (`bits`)."""
    if fmt in PACKED:
        n = torch.stack([q & 0xF, q >> 4], -1).reshape(-1, g)
        if fmt == "nvfp4":
            return (e2m1_value(n) * (bits.view(torch.float8_e4m3fn).float() * gs).view(-1, 1)).reshape(-1)
        n = n.to(torch.int32)
        vals = torch.where(n >= 8, n - 16, n).float() if fmt == "int4" else n.float() - bits.float().view(-1, 1)
        return (vals * scale.view(-1, 1)).reshape(-1)
    vals = q.view(torch.int8).float() if fmt == "int8" else q.view(_fp8(fmt)).float()
    factor = bits.view(torch.float8_e8m0fnu).float() if fmt.startswith("mx") else scale
    return (vals.view(-1, g) * factor.view(-1, 1)).reshape(-1)


# ---- torch baselines of ce.quantize / ce.dequantize ---------------------------------

def _scale_out(scale, fmt):
    return scale.view(torch.uint8) if fmt.startswith("mx") or fmt == "nvfp4" else scale


def _rows(t, idx):
    """The rows of `t` an index selects (all of them without one)."""
    return t if idx is None else t.index_select(0, idx.to(t.device))


def _write_rows(t, idx, v):
    if idx is None:
        t.copy_(v.view(t.shape).to(t.dtype), non_blocking=True)
    else:
        t.index_copy_(0, idx.to(t.device), v.view((len(idx),) + tuple(t.shape[1:])).to(t.device, t.dtype))


def native_quantize(x, q, scale, zero, fmt, g, eps, gs, x_index=None, index=None):
    codes, s, bits = quantize(_rows(x, x_index), g, fmt, eps, gs)
    _write_rows(q.view(torch.uint8), index, codes)
    if s is not None:
        _write_rows(scale, index, s)
    if fmt == "uint4":
        _write_rows(zero, index, bits)
    elif bits is not None:
        _write_rows(_scale_out(scale, fmt), index, bits)


def native_dequantize(q, scale, zero, out, fmt, g, gs, x_index=None, index=None):
    codes = _rows(q.view(torch.uint8), index).reshape(-1)
    sc = _rows(_scale_out(scale, fmt), index)
    s = sc.reshape(-1) if scale.dtype == torch.float32 else None
    bits = _rows(zero, index).reshape(-1) if fmt == "uint4" else (None if s is not None else sc.reshape(-1))
    _write_rows(out, x_index, dequantize(codes, s, bits, g, fmt, gs))
