"""Box units: one scale per bh x bw block of the last two dimensions (block
FP8 / INT8, as DeepSeek-V3 stores weights), bit for bit against the recipes
in copyengine/_quant.py, on the host and on the GPU."""

import pytest

import copyengine as ce
from copyengine import _quant

torch = pytest.importorskip("torch")
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
Q_DTYPE = {"fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2, "int8": torch.int8}


def data(shape, dtype, dev, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g) * torch.exp(torch.randn(*shape[:-1], 1, generator=g) * 3)
    x.view(-1)[:3] = torch.tensor([float("nan"), float("inf"), 0.0])
    return x.to(dtype).to(dev)


def check_codes(fmt, got, want):
    got, want = got.cpu().view(torch.uint8).reshape(-1), want.reshape(-1)
    if fmt == "int8":
        assert torch.equal(got, want)
        return
    nan = got.view(Q_DTYPE[fmt]).float().isnan()  # NaN patterns differ between torch's CPU and CUDA casts
    assert torch.equal(nan, want.view(Q_DTYPE[fmt]).float().isnan())
    assert torch.equal(got[~nan], want[~nan])


def same_float(a, b):
    a, b = a.cpu().float(), b.cpu().float()
    nan = a.isnan()
    return torch.equal(nan, b.isnan()) and torch.equal(a[~nan], b[~nan])


def test_boxes_round_trip():
    x = torch.arange(2 * 8 * 12).reshape(2, 8, 12)
    u = _quant.boxes(x, 4, 3)
    assert u.shape == (2 * 2 * 4, 12)
    assert torch.equal(u[0], x[0, :4, :3].reshape(-1))
    assert torch.equal(_quant.unboxes(u, x.shape, 4, 3), x)


def engines():
    return [ce.Engine(disable=True), ce.Engine(tune_trials=1, tune_topk=4)]


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2", "int8"])
@pytest.mark.parametrize("block,shape", [((128, 128), (2, 256, 384)), ((4, 32), (3, 8, 64)), ((16, 8), (32, 24))])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_block_quantize_matches_recipe(dev, fmt, block, shape, dtype):
    bh, bw = block
    x = data(shape, dtype, dev, seed=bh + bw)
    *lead, M, N = shape
    for e in engines():
        for _ in range(4):
            q = torch.zeros(shape, dtype=Q_DTYPE[fmt], device=dev)
            scale = torch.zeros(*lead, M // bh, N // bw, device=dev)
            e.quantize(x, q, scale, fmt=fmt, block=block)
            if dev == "cuda":
                torch.cuda.synchronize()
            codes, s = _quant.quantize_block(x.cpu(), block, fmt)
            check_codes(fmt, q, codes)
            assert same_float(scale, s)
            back = torch.zeros(shape, dtype=dtype, device=dev)
            e.dequantize(q, scale, back, fmt=fmt, block=block)
            if dev == "cuda":
                torch.cuda.synchronize()
            want = _quant.dequantize_block(q.cpu().view(torch.uint8), scale.cpu(), block, fmt).to(dtype)
            assert same_float(back, want)
            q2 = torch.zeros_like(q)
            e.quantize(x, q2, scale, fmt=fmt, block=block, static=True)
            if dev == "cuda":
                torch.cuda.synchronize()
            check_codes(fmt, q2, q.view(torch.uint8).cpu())  # NaN for NaN (signs may differ)


@pytest.mark.parametrize("dev", DEVS)
def test_block_strided_and_indexed(dev):
    # A transposed weight (no vector accesses), and pages of a stack gathered by index.
    fmt, block = "fp8_e4m3", (8, 16)
    w = data((64, 48), torch.bfloat16, dev, seed=3).t()  # [48, 64], strided
    q = torch.zeros(48, 64, dtype=torch.float8_e4m3fn, device=dev)
    s = torch.zeros(6, 4, device=dev)
    ce.Engine(prior_only=True).quantize(w, q, s, fmt=fmt, block=block)
    if dev == "cuda":
        torch.cuda.synchronize()
    codes, sc = _quant.quantize_block(w.cpu(), block, fmt)
    check_codes(fmt, q, codes)
    assert same_float(s, sc)
    stack = data((10, 16, 32), torch.float32, dev, seed=4)
    qs = torch.zeros(6, 16, 32, dtype=torch.float8_e4m3fn, device=dev)
    ss = torch.zeros(6, 2, 2, device=dev)
    pages, slots = [7, 2, 9], [5, 0, 3]
    ix = torch.tensor(pages, device=dev) if dev == "cuda" else pages
    e = ce.Engine(prior_only=True)
    e.quantize(stack, qs, ss, fmt=fmt, block=block, x_index=ix, index=slots)
    if dev == "cuda":
        torch.cuda.synchronize()
    codes, sc = _quant.quantize_block(stack[pages].cpu(), block, fmt)
    check_codes(fmt, qs.view(torch.uint8)[slots], codes)  # byte view: old torch cannot index fp8 on the CPU
    assert same_float(ss[slots], sc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_block_into_pinned_host():
    fmt, block = "int8", (32, 64)
    x = data((4, 256, 256), torch.bfloat16, "cuda", seed=5)
    q = torch.zeros(4, 256, 256, dtype=torch.int8).pin_memory()
    s = torch.zeros(4, 8, 4).pin_memory()
    seen = set()
    e = ce.Engine(tune_trials=1, tune_topk=4)
    for _ in range(8):
        e.quantize(x, q, s, fmt=fmt, block=block)
        e.synchronize()
        seen.update(e.stats()["last"]["strategies"])
        codes, sc = _quant.quantize_block(x.cpu(), block, fmt)
        check_codes(fmt, q, codes)
        assert same_float(s, sc)
    assert "tile_staged" in seen and "tile" in seen


def test_block_rejects_other_formats_and_shapes():
    x = torch.randn(8, 8)
    with pytest.raises(ValueError):
        ce.quantize(x, torch.zeros(8, 8, dtype=torch.uint8), torch.zeros(2, 2), fmt="nvfp4", block=(4, 4))
    with pytest.raises(ValueError):
        ce.quantize(x, torch.zeros(8, 8, dtype=torch.int8), torch.zeros(2, 2), fmt="int8", block=(3, 4))
    with pytest.raises(ValueError):
        ce.quantize(x, torch.zeros(8, 8, dtype=torch.int8), torch.zeros(2, 3), fmt="int8", block=(4, 4))


@pytest.mark.skipif(not torch.cuda.is_available() or not ce._compiled.available(), reason="needs CUDA and torch ops")
def test_block_quantize_compiled(monkeypatch):
    monkeypatch.setenv("CE_COMPILE_BASELINE", "0")
    fmt, block = "fp8_e4m3", (128, 128)
    w = data((256, 512), torch.bfloat16, "cuda", seed=6)
    q = torch.zeros(256, 512, dtype=torch.float8_e4m3fn, device="cuda")
    s = torch.zeros(2, 4, device="cuda")

    def f(w):
        ce.quantize(w, q, s, fmt=fmt, block=block)
        return s.sum()

    torch._dynamo.reset()
    assert torch._dynamo.explain(f)(w).graph_break_count == 0
    torch._dynamo.reset()
    torch.compile(f)(w)
    torch.cuda.synchronize()
    codes, sc = _quant.quantize_block(w.cpu(), block, fmt)
    check_codes(fmt, q, codes)
    assert same_float(s, sc)
