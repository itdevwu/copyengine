"""FP8 and packed FP4: copyengine must give torch's exact bits, on host and GPU."""

import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

F8 = [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu]
WIDE = [torch.float32, torch.float16, torch.bfloat16, torch.float64]
BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def bits(t):
    return t.contiguous().view(BITS[t.element_size()]).cpu()


def same(got, want):
    """torch's bits for every non-NaN value; NaN stays NaN. (torch's own NaN
    payloads differ between its CPU and CUDA conversions, so none is copied.)"""
    if want.dtype == torch.float4_e2m1fn_x2:
        return torch.equal(bits(got), bits(want))
    nan_w, nan_g = want.float().isnan().cpu(), got.float().isnan().cpu()
    return torch.equal(nan_w, nan_g) and torch.equal(bits(got)[~nan_w], bits(want)[~nan_w])


def rows(t, ids):
    """t[ids] by bytes (older torch cannot index fp8 tensors on the CPU)."""
    return t.view(torch.uint8)[torch.tensor(ids, device=t.device)].view(t.dtype)


def ce_copy(dst_dtype, src, fn=None):
    out = torch.empty(src.shape, dtype=dst_dtype, device=src.device)
    ce.Engine(prior_only=True).copy_(out, src, fn=fn)  # ce's own kernels, never torch's copy
    return out


def every_f8(dt, dev):
    return torch.arange(256, dtype=torch.int32).to(torch.uint8).view(dt).to(dev)


def encode_inputs():
    """float32 values that exercise every fp8 rounding decision."""
    allf16 = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.float16).float()
    allbf16 = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    mids = []
    for dt in F8:  # exact ties between neighbouring fp8 values, and just around them
        v = every_f8(dt, "cpu").double()
        v = v[torch.isfinite(v)].unique().sort().values
        m = ((v[1:] + v[:-1]) / 2).float()
        mids += [m, m.nextafter(torch.full_like(m, float("inf"))), m.nextafter(torch.full_like(m, -float("inf")))]
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(1 << 16, generator=g) * 2.0 ** torch.randint(-30, 30, (1 << 16,), generator=g).float()
    specials = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 448.0, 464.0, 463.99, 480.0,
                             57344.0, 61440.0, 65536.0, 2.0 ** -6, 2.0 ** -9, 2.0 ** -10, 2.0 ** -14, 2.0 ** -16,
                             2.0 ** -17, 2.0 ** -127, 2.0 ** -126, 2.0 ** 127, 3.4e38, 1e-45])
    x = torch.cat([allf16, allbf16, *mids, rnd, specials, -specials])
    return x[torch.isfinite(x) | torch.isnan(x) | torch.isinf(x)]


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("f8", F8)
@pytest.mark.parametrize("wide", WIDE)
def test_decode_every_fp8_value(dev, f8, wide):
    x = every_f8(f8, dev)
    assert same(ce_copy(wide, x), x.to(wide))


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("f8", F8)
@pytest.mark.parametrize("wide", WIDE)
def test_encode_matches_torch(dev, f8, wide):
    x = encode_inputs().to(wide).to(dev)
    assert same(ce_copy(f8, x), x.to(f8))


@pytest.mark.parametrize("dev", DEVS)
def test_fp8_to_fp8_and_ints(dev):
    for a in F8:
        for b in F8:
            x = every_f8(a, dev)
            assert same(ce_copy(b, x), x.to(b)), (a, b)
    ints = torch.tensor([0, 1, -1, 7, 448, 449, 1000, 57344, 2 ** 20, -(2 ** 20), 2 ** 31 - 1], device=dev)
    for it in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        x = ints.clamp(torch.iinfo(it).min, torch.iinfo(it).max).to(it)
        for b in F8:
            assert same(ce_copy(b, x), x.to(b)), (it, b)


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("f8", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_quantize_and_dequantize_with_scale(dev, f8):
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(4096, generator=g) * 300).to(torch.bfloat16).to(dev)
    lim = float(torch.finfo(f8).max)
    s = 1.7
    q = ce_copy(f8, x, fn=ce.Elementwise(scale=s, clamp=(-lim, lim)))  # saturating quantization
    want = (x.float() * s).clamp(-lim, lim).to(f8)
    assert same(q, want)
    d = ce_copy(torch.bfloat16, q, fn=ce.Elementwise(scale=1 / s))
    assert same(d, (q.float() * (1 / s)).to(torch.bfloat16))


@pytest.mark.parametrize("dev", DEVS)
def test_strided_and_scattered_fp8_copies(dev):
    cache = torch.randn(64, 8, 32).to(torch.float8_e4m3fn).to(dev)
    ids = torch.randperm(64)[:20].tolist()
    out = torch.empty(20, 32, 8, dtype=torch.bfloat16, device=dev)
    ce.Engine(prior_only=True).copy_(out, ce.lazy(cache).index_select(0, ids).permute(0, 2, 1))
    want = rows(cache, ids).permute(0, 2, 1).to(torch.bfloat16)
    assert same(out, want)


@pytest.mark.parametrize("dev", DEVS)
def test_packed_fp4_moves_as_bytes_only(dev):
    raw = torch.randint(0, 256, (16, 32), dtype=torch.uint8)
    src = raw.view(torch.float4_e2m1fn_x2).to(dev)
    dst = torch.empty(32, 16, dtype=torch.float4_e2m1fn_x2, device=dev)
    ce.Engine(prior_only=True).copy_(dst, src.t())
    assert torch.equal(dst.view(torch.uint8).cpu(), raw.t())
    with pytest.raises(ce.Unsupported):
        ce.Engine(prior_only=True).copy_(torch.empty(16, 32, dtype=torch.uint8, device=dev), src)


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("intercept", ["mode", "shim"])
def test_intercepted_fp8_copies(dev, intercept):
    if intercept == "shim" and ce._shim.lib is None:
        pytest.skip(ce._shim.error)
    src = torch.randn(8, 16, 64).to(dev)
    cache = torch.zeros(8, 40, 64, dtype=torch.float8_e4m3fn, device=dev)
    ids = torch.randperm(40)[:16].tolist()
    e = ce.Engine(prior_only=True, _intercept=intercept)
    with e.scope():
        for l in range(8):
            for i, b in enumerate(ids):
                cache[l, b].copy_(src[l, i])
    assert e.stats()["last"]["ops"] == 8 * 16
    assert same(rows(cache.transpose(0, 1), ids).transpose(0, 1), src.to(torch.float8_e4m3fn))


_OVERFLOW_SCRIPT = r"""
import sys, torch, copyengine as ce
dev, sat = sys.argv[1], sys.argv[2] == "1"
assert ce._core._f8e4m3_saturate() == sat
x = torch.tensor([447.0, 448.0, 463.99, 464.0, 465.0, 480.0, 1e4, 3.4e38, float("inf"), float("nan"), 1.0, 0.0])
x = torch.cat([x, -x]).to(dev)
for wide in (torch.float32, torch.float16, torch.bfloat16):
    xs = x.to(wide)
    # The expected bits: torch's conversion where both modes agree; beyond the
    # format (|x| > 464, 464 rounding to even 448) NaN or +-448 by the mode.
    xc = xs.cpu()
    want = xc.to(torch.float8_e4m3fn).view(torch.uint8)
    beyond = ~xc.isnan() & (xc.float().abs() > 464)
    sign = torch.where(torch.signbit(xc.float()), 0x80, 0).to(torch.uint8)
    want = torch.where(beyond, sign | (0x7E if sat else 0x7F), want)
    out = torch.empty(xs.shape, dtype=torch.float8_e4m3fn, device=dev)
    ce.copy_(out, xs)
    if dev == "cuda":
        torch.cuda.synchronize()
    got = out.view(torch.uint8).cpu()
    nan = (want & 0x7F) == 0x7F
    assert torch.equal(got[~nan], want[~nan]) and torch.equal((got[nan] & 0x7F), want[nan] & 0x7F), (wide, got, want)
print("ok")
"""


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("sat", ["0", "1"])
def test_e4m3_overflow_follows_torch_mode(dev, sat, tmp_path):
    # torch 2.13 saturates float -> float8_e4m3fn beyond the format (+-448);
    # earlier releases return NaN. Copies follow the installed torch, on the
    # host and in the kernels; CE_F8E4M3_SATURATE forces either mode here.
    import os, subprocess, sys
    env = dict(os.environ, CE_F8E4M3_SATURATE=sat)
    r = subprocess.run([sys.executable, "-c", _OVERFLOW_SCRIPT, dev, sat], env=env, capture_output=True, text=True,
                       cwd=tmp_path, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout + r.stderr


def test_e4m3_overflow_mode_matches_installed_torch():
    import os
    if "CE_F8E4M3_SATURATE" in os.environ:
        pytest.skip("mode forced")
    sat = torch.tensor([1000.0]).to(torch.float8_e4m3fn).view(torch.uint8).item() == 0x7E
    assert ce._core._f8e4m3_saturate() == sat
