"""The quantization catalog's host implementation (src/core/quant.hpp) against
the torch recipes of docs/design/v0.4.md, bit for bit (NaN for NaN)."""

import numpy as np
import pytest

from copyengine import _core, _quant

torch = pytest.importorskip("torch")

FMT = {"fp8_e4m3": 0, "fp8_e5m2": 1, "int8": 2, "mxfp8_e4m3": 3, "mxfp8_e5m2": 4, "nvfp4": 5, "int4": 6, "uint4": 7}
QMAX = {"fp8_e4m3": 448.0, "fp8_e5m2": 57344.0, "int8": 127.0, "mxfp8_e4m3": 448.0, "mxfp8_e5m2": 57344.0,
        "int4": 7.0, "uint4": 15.0}
FP8 = {"fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2,
       "mxfp8_e4m3": torch.float8_e4m3fn, "mxfp8_e5m2": torch.float8_e5m2}
EPS = 1e-12


def _bits_f32(b):
    return torch.tensor(b, dtype=torch.int32).view(torch.float32)


# ---- the torch recipes (copyengine._quant: the one copy of each) ---------------------

def recipe(x, g, fmt, gs=1.0):
    return _quant.quantize(x, g, fmt, EPS, gs)


def recipe_dequant(q, scale, bits, g, fmt, gs=1.0):
    return _quant.dequantize(q, scale, bits, g, fmt, gs)


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
    if fmt in ("int8", "nvfp4", "int4", "uint4"):
        return q
    v = q.view(FP8[fmt]).float()
    return torch.where(v.isnan(), torch.tensor(0xFF, dtype=torch.uint8), q)


@pytest.mark.parametrize("fmt,g", [("fp8_e4m3", 128), ("fp8_e5m2", 64), ("int8", 128), ("fp8_e4m3", 16),
                                   ("mxfp8_e4m3", 32), ("mxfp8_e5m2", 32), ("nvfp4", 16), ("int4", 32),
                                   ("int4", 128), ("uint4", 32), ("uint4", 128)])
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


@pytest.mark.parametrize("fmt,g", [("fp8_e4m3", 128), ("int8", 32), ("mxfp8_e5m2", 32), ("nvfp4", 16),
                                   ("int4", 32), ("uint4", 64)])
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
    if fmt in ("int8", "int4"):
        return np.arange(int(QMAX[fmt]) + 1, dtype=np.float32)
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
        scale = np.float32(max(amax, np.float32(EPS)) * np.float32(1 / QMAX[fmt]))
        m = rng.choice(mids, g - 1) * rng.choice([-1, 1], g - 1)
        out.append(np.r_[amax, (m * scale).astype(np.float32)].astype(np.float32))
    return np.concatenate(out)


@pytest.mark.parametrize("fmt,g", [("fp8_e4m3", 32), ("fp8_e5m2", 32), ("int8", 32), ("int4", 32)])
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


def test_uint4_ties():
    # Units spanning [-a, b]: x / scale + zero lands on half-integers.
    rng = np.random.default_rng(3)
    out = []
    for _ in range(400):
        a, b = (np.float32(10.0 ** rng.uniform(-3, 3)) for _ in range(2))
        scale = np.float32((np.float32(b) + np.float32(a)) * np.float32(1 / 15))
        k = rng.integers(0, 15, 30) + 0.5
        z = np.float32(np.round(a / scale))
        vals = ((k - z) * scale).astype(np.float32)
        out.append(np.r_[-a, b, vals].astype(np.float32))
    x = np.concatenate(out)
    q, scale, bits = host("uint4", x, 32)
    rq, rscale, rbits = recipe(torch.from_numpy(x), 32, "uint4")
    assert same(q, rq) and same(scale, rscale) and same(bits, rbits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("fmt,g", [("fp8_e4m3", 128), ("int8", 32), ("int4", 32), ("uint4", 32),
                                   ("mxfp8_e4m3", 32), ("nvfp4", 16)])
def test_recipes_agree_across_devices(fmt, g):
    # The recipe is the torch baseline on the GPU too: it must give the same
    # bits there (torch divides by a scalar differently on the GPU).
    x = data(g, seed=11)
    gs = 0.01 if fmt == "nvfp4" else 1.0
    cpu = recipe(torch.from_numpy(x), g, fmt, gs)
    gpu = [None if t is None else t.cpu() for t in recipe(torch.from_numpy(x).cuda(), g, fmt, gs)]
    assert same(fp8_nan_positions(cpu[0], fmt), fp8_nan_positions(gpu[0], fmt))
    for a, b in zip(cpu[1:], gpu[1:]):
        assert (a is None) == (b is None) and (a is None or same(a, b))
