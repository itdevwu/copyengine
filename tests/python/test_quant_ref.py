"""The quantization catalog's host implementation (src/core/quant.hpp) against
the torch recipes of docs/design/v0.4.md, bit for bit (NaN for NaN)."""

import numpy as np
import pytest

from copyengine import _core

torch = pytest.importorskip("torch")

FMT = {"fp8_e4m3": 0, "fp8_e5m2": 1, "int8": 2, "mxfp8_e4m3": 3, "mxfp8_e5m2": 4, "nvfp4": 5}
QMAX = {"fp8_e4m3": 448.0, "fp8_e5m2": 57344.0, "int8": 127.0, "mxfp8_e4m3": 448.0, "mxfp8_e5m2": 57344.0}
FP8 = {"fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2,
       "mxfp8_e4m3": torch.float8_e4m3fn, "mxfp8_e5m2": torch.float8_e5m2}
EPS = 1e-12


def _bits_f32(b):
    return torch.tensor(b, dtype=torch.int32).view(torch.float32)


# ---- torch recipes (the design document's, written once) ----------------------------

def recipe_scaled(x, g, fmt, eps=EPS):
    xf = x.view(-1, g)
    amax = xf.abs().amax(-1, keepdim=True)
    scale = amax.clamp(min=eps) / QMAX[fmt]
    v = xf / scale
    if fmt == "int8":
        q = torch.where(v.isnan(), 0.0, torch.round(v).clamp(-127, 127)).to(torch.int8).view(torch.uint8)
    else:
        q = v.clamp(-QMAX[fmt], QMAX[fmt]).to(FP8[fmt]).view(torch.uint8)
    return q.reshape(-1), scale.reshape(-1), None


def recipe_mx(x, g, fmt):
    emax = 8 if fmt == "mxfp8_e4m3" else 15
    xf = x.view(-1, g)
    amax = xf.abs().amax(-1, keepdim=True)
    _, e = torch.frexp(amax)
    shared = (e - 1 - emax).clamp(-127, 127)
    shared = torch.where(amax == 0, -127, shared)
    finite = amax.isfinite()
    sf = torch.where(shared >= -126, ((shared + 127).to(torch.int32) << 23).view(torch.float32),
                     _bits_f32(0x00400000))
    sf = torch.where(finite, sf, float("nan"))
    q = (xf / sf).clamp(-QMAX[fmt], QMAX[fmt]).to(FP8[fmt]).view(torch.uint8)
    bits = torch.where(finite, shared + 127, 255).to(torch.uint8)
    return q.reshape(-1), None, bits.reshape(-1)


def e2m1(v):
    m = v.abs()
    code = sum((m > 0.25, m >= 0.75, m > 1.25, m >= 1.75, m > 2.5, m >= 3.5, m > 5.0), torch.zeros_like(m)).to(
        torch.uint8)
    code = code | (torch.signbit(v).to(torch.uint8) << 3)
    return torch.where(v.isnan(), 0, code).to(torch.uint8)


def e2m1_value(code):
    mag = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[(code & 7).long()]
    return torch.where((code & 8) != 0, -mag, mag)


def recipe_nvfp4(x, g, gs):
    xf = x.view(-1, g)
    amax = xf.abs().amax(-1, keepdim=True)
    sv = (amax / 6 / gs).clamp(max=448)
    s = sv.to(torch.float8_e4m3fn)
    sf = s.float()
    v = (xf / (sf * gs)).clamp(-6, 6)
    code = torch.where(sf == 0, 0, e2m1(v)).to(torch.uint8)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed.reshape(-1), None, s.view(torch.uint8).reshape(-1)


def recipe(x, g, fmt, gs=1.0):
    if fmt in ("fp8_e4m3", "fp8_e5m2", "int8"):
        return recipe_scaled(x, g, fmt)
    if fmt.startswith("mx"):
        return recipe_mx(x, g, fmt)
    return recipe_nvfp4(x, g, gs)


def recipe_dequant(q, scale, bits, g, fmt, gs=1.0):
    if fmt == "nvfp4":
        codes = torch.stack([q & 0xF, q >> 4], -1).reshape(-1, g)
        return (e2m1_value(codes) * (bits.view(torch.float8_e4m3fn).float() * gs).view(-1, 1)).reshape(-1)
    if fmt == "int8":
        vals = q.view(torch.int8).float()
    else:
        vals = q.view(FP8[fmt]).float()
    if fmt.startswith("mx"):
        factor = bits.view(torch.float8_e8m0fnu).float()
    else:
        factor = scale
    return (vals.view(-1, g) * factor.view(-1, 1)).reshape(-1)


# ---- data ----------------------------------------------------------------------

def data(g, units=64, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(units * g).astype(np.float32)
    x *= np.repeat(10.0 ** rng.uniform(-30, 30, units), g).astype(np.float32)  # every magnitude
    edges = [
        np.zeros(g),                                         # an all-zero unit
        np.full(g, 1e-45), np.full(g, 3e-39),                # subnormal amax
        np.full(g, 448.0), np.full(g, 464.0), np.full(g, 1e30),
        np.r_[np.nan, np.ones(g - 1)], np.r_[np.inf, np.ones(g - 1)], np.r_[-np.inf, np.zeros(g - 1)],
        np.linspace(-6.5, 6.5, g), np.r_[np.full(g // 2, -0.0), np.full(g - g // 2, 0.25)],
    ]
    return np.concatenate([x] + [e.astype(np.float32) for e in edges]).astype(np.float32)


def same(a, b):
    """Bit for bit; float tensors: NaN for NaN."""
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    if a.dtype == torch.float32:
        nan = a.isnan()
        return bool(torch.equal(nan, b.isnan()) and torch.equal(a[~nan].view(torch.int32), b[~nan].view(torch.int32)))
    return bool(torch.equal(a, b))


def host(fmt, x, g, gs=1.0):
    q, scale, bits = _core._quant_ref(FMT[fmt], x, g, EPS, gs)
    return (torch.frombuffer(bytearray(q), dtype=torch.uint8), torch.frombuffer(bytearray(scale), dtype=torch.float32),
            torch.frombuffer(bytearray(bits), dtype=torch.uint8))


def fp8_nan_positions(q, fmt):
    """fp8 NaN bit patterns differ between torch's CPU and CUDA paths: compare where NaNs are."""
    if fmt == "int8" or fmt == "nvfp4":
        return q
    v = q.view(FP8[fmt]).float()
    return torch.where(v.isnan(), torch.tensor(0xFF, dtype=torch.uint8), q)


@pytest.mark.parametrize("fmt,g", [("fp8_e4m3", 128), ("fp8_e5m2", 64), ("int8", 128), ("fp8_e4m3", 16),
                                   ("mxfp8_e4m3", 32), ("mxfp8_e5m2", 32), ("nvfp4", 16)])
def test_quantize_matches_the_recipe(fmt, g):
    for seed in range(3):
        x = data(g, seed=seed)
        gs = 1.0 if fmt != "nvfp4" else float(np.abs(x[np.isfinite(x)]).max() / (6 * 448))
        q, scale, bits = host(fmt, x, g, gs)
        rq, rscale, rbits = recipe(torch.from_numpy(x), g, fmt, gs)
        assert same(fp8_nan_positions(q, fmt), fp8_nan_positions(rq, fmt)), (fmt, seed)
        if rscale is not None:
            assert same(scale, rscale), (fmt, seed)
        if rbits is not None:
            assert same(bits, rbits), (fmt, seed)


@pytest.mark.parametrize("fmt,g", [("fp8_e4m3", 128), ("int8", 32), ("mxfp8_e5m2", 32), ("nvfp4", 16)])
def test_dequantize_matches_the_recipe(fmt, g):
    x = data(g, seed=7)
    gs = 1.0 if fmt != "nvfp4" else 0.01
    q, scale, bits = host(fmt, x, g, gs)
    y = torch.frombuffer(bytearray(_core._dequant_ref(FMT[fmt], q.numpy().tobytes(), scale.numpy().tobytes(),
                                                      bits.numpy().tobytes(), g, gs)), dtype=torch.float32)
    assert same(y, recipe_dequant(q, scale, bits, g, fmt, gs))


# ---- ties: where rounding direction and the order of operations show ----------------

def _grid(fmt):
    """The non-negative values of a format, ascending."""
    if fmt == "int8":
        return np.arange(128, dtype=np.float32)
    v = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(FP8[fmt]).float().numpy()
    v = v[np.isfinite(v) & (v >= 0)]
    return np.unique(v[v <= QMAX[fmt]])


def tie_data(fmt, g, units=400, seed=0):
    """Units whose values sit on the midpoints of the format's grid, scaled by
    the unit's own scale: x / scale lands on a tie (or one ulp off it, which is
    where x / scale and x * (1 / scale) part)."""
    rng = np.random.default_rng(seed)
    grid = _grid(fmt)
    mids = ((grid[1:] + grid[:-1]) / 2).astype(np.float32)
    out = []
    for _ in range(units):
        amax = np.float32(10.0 ** rng.uniform(-3, 3))
        scale = np.float32(max(amax, np.float32(EPS)) / np.float32(QMAX[fmt]))
        m = rng.choice(mids, g - 1) * rng.choice([-1, 1], g - 1)
        out.append(np.r_[amax, (m * scale).astype(np.float32)].astype(np.float32))
    return np.concatenate(out)


@pytest.mark.parametrize("fmt,g", [("fp8_e4m3", 32), ("fp8_e5m2", 32), ("int8", 32)])
def test_scaled_ties(fmt, g):
    x = tie_data(fmt, g)
    q, scale, _ = host(fmt, x, g)
    rq, rscale, _ = recipe(torch.from_numpy(x), g, fmt)
    assert same(q, rq) and same(scale, rscale)


def test_nvfp4_ties():
    # amax 6 and gs 1: the E4M3 scale is 1, so x / (scale * gs) is x itself,
    # and every E2M1 tie is hit exactly.
    ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 0.0, 0.5, 1.5, 3.0, 4.0, -0.25, -0.75, -1.25]
    x = np.array([6.0] + ties[:15] + [-6.0] + [-t for t in ties[:15]], dtype=np.float32)
    q, _, bits = host("nvfp4", x, 16, 1.0)
    rq, _, rbits = recipe(torch.from_numpy(x), 16, "nvfp4", 1.0)
    assert same(q, rq) and same(bits, rbits)
