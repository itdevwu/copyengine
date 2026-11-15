"""ce.quantize / ce.dequantize against the torch recipes (copyengine._quant),
bit for bit, through copyengine's own strategy and through the torch baseline."""

import pytest

import copyengine as ce
from copyengine import _quant

torch = pytest.importorskip("torch")
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

FORMATS = [("fp8_e4m3", 128), ("fp8_e5m2", 64), ("int8", 32), ("int4", 32), ("uint4", 64),
           ("mxfp8_e4m3", 32), ("mxfp8_e5m2", 32), ("nvfp4", 16)]
Q_DTYPE = {"fp8_e4m3": torch.float8_e4m3fn, "mxfp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2,
           "mxfp8_e5m2": torch.float8_e5m2, "int8": torch.int8}
S_DTYPE = {"mxfp8_e4m3": torch.float8_e8m0fnu, "mxfp8_e5m2": torch.float8_e8m0fnu, "nvfp4": torch.float8_e4m3fn}


def outputs(fmt, lead, D, g, dev):
    packed = fmt in _quant.PACKED
    q = torch.zeros(*lead, D // 2 if packed else D, dtype=Q_DTYPE.get(fmt, torch.uint8), device=dev)
    scale = torch.zeros(*lead, D // g, dtype=S_DTYPE.get(fmt, torch.float32), device=dev)
    zero = torch.zeros(*lead, D // g, dtype=torch.uint8, device=dev) if fmt == "uint4" else None
    return q, scale, zero


def data(lead, D, dtype, dev, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*lead, D, generator=g) * torch.exp(torch.randn(*lead, 1, generator=g) * 4)
    x.view(-1)[:3] = torch.tensor([float("nan"), float("inf"), 0.0])
    return x.to(dtype).to(dev)


def rows(t, idx):
    """t[idx] through a byte view: old torch cannot index fp8 tensors on the CPU."""
    if t.element_size() == 1:
        return t.view(torch.uint8)[idx.to(t.device)].view(t.dtype)
    return t[idx.to(t.device)]


def engines():
    # The torch baseline only, and ce's strategies while exploring (the
    # recipe's own bits either way).
    return [ce.Engine(disable=True), ce.Engine(tune_trials=1, tune_topk=4)]


def check(fmt, g, x, q, scale, zero, gs=1.0):
    codes, s, bits = _quant.quantize(x.cpu(), g, fmt, gs=gs)
    got = q.cpu().view(torch.uint8).reshape(-1)
    if fmt in Q_DTYPE and fmt != "int8":  # fp8 NaN patterns differ between torch's CPU and CUDA casts: positions
        nan = got.view(Q_DTYPE[fmt]).float().isnan()
        assert torch.equal(nan, codes.view(Q_DTYPE[fmt]).float().isnan())
        assert torch.equal(got[~nan], codes[~nan])
    else:
        assert torch.equal(got, codes)
    if scale.dtype == torch.float32:
        a, b = scale.cpu().reshape(-1), s
        nan = a.isnan()
        assert torch.equal(nan, b.isnan()) and torch.equal(a[~nan], b[~nan])
    else:
        assert torch.equal(scale.cpu().view(torch.uint8).reshape(-1), bits)
    if zero is not None:
        assert torch.equal(zero.cpu().reshape(-1), bits)


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("fmt,g", FORMATS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_quantize(dev, fmt, g, dtype):
    lead, D = (3, 5), 4 * g
    x = data(lead, D, dtype, dev)
    gs = 0.02 if fmt == "nvfp4" else 1.0
    for e in engines():
        for _ in range(6):
            q, scale, zero = outputs(fmt, lead, D, g, dev)
            e.quantize(x, q, scale, zero, fmt=fmt, group=g, global_scale=gs)
            check(fmt, g, x, q, scale, zero, gs)


@pytest.mark.parametrize("fmt,g", FORMATS)
def test_strided_rows_and_dequantize(fmt, g):
    D = 2 * g
    x = data((6, 4), D, torch.float32, "cpu").transpose(0, 1)  # strided rows
    gs = 0.02 if fmt == "nvfp4" else 1.0
    e = ce.Engine()
    q, scale, zero = outputs(fmt, (4, 6), D, g, "cpu")
    e.quantize(x, q, scale, zero, fmt=fmt, group=g, global_scale=gs)
    check(fmt, g, x, q, scale, zero, gs)
    out = torch.empty(4, 6, D, dtype=torch.bfloat16)
    e.dequantize(q, scale, out, zero, fmt=fmt, group=g, global_scale=gs)
    codes, s, bits = _quant.quantize(x, g, fmt, gs=gs)
    want = _quant.dequantize(codes, s, bits, g, fmt, gs).view(out.shape).to(out.dtype)
    nan = out.isnan()
    assert torch.equal(nan, want.isnan()) and torch.equal(out[~nan], want[~nan])


def test_layers_quantized_in_one_scope_are_one_group():
    e = ce.Engine(prior_only=True)
    xs = [data((8,), 256, torch.float32, "cpu", seed=k) for k in range(5)]
    outs = [outputs("fp8_e4m3", (8,), 256, 128, "cpu") for _ in xs]
    with e.scope():
        for x, (q, s, _) in zip(xs, outs):
            e.quantize(x, q, s, fmt="fp8_e4m3", group=128)
    last = e.stats()["last"]
    assert last["ops"] == 5 and last["groups"] == 1 and last["strategies"] == ["tile"]
    for x, (q, s, z) in zip(xs, outs):
        check("fp8_e4m3", 128, x, q, s, z)


def test_bad_shapes_are_rejected():
    x = torch.randn(4, 100)
    q, s, _ = outputs("fp8_e4m3", (4,), 128, 128, "cpu")
    with pytest.raises(ValueError):
        ce.quantize(x, q, s, fmt="fp8_e4m3", group=128)
    with pytest.raises(ValueError):
        ce.quantize(torch.randn(4, 64), *outputs("mxfp8_e4m3", (4,), 64, 32, "cpu")[:2], fmt="mxfp8_e4m3", group=16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("fmt,g", FORMATS + [("fp8_e4m3", 4096), ("uint4", 1024)])  # long units: many values per lane
def test_gpu_strided_and_dequantize(fmt, g):
    D = 2 * g
    x = data((6, 4), D, torch.float16, "cuda").transpose(0, 1)
    gs = 0.02 if fmt == "nvfp4" else 1.0
    e = ce.Engine(prior_only=True)
    q, scale, zero = outputs(fmt, (4, 6), D, g, "cuda")
    e.quantize(x, q, scale, zero, fmt=fmt, group=g, global_scale=gs)
    assert e.stats()["last"]["strategies"] == ["tile"]
    torch.cuda.synchronize()
    check(fmt, g, x, q, scale, zero, gs)
    out = torch.empty(4, 6, D, dtype=torch.float32, device="cuda")
    e.dequantize(q, scale, out, zero, fmt=fmt, group=g, global_scale=gs)
    torch.cuda.synchronize()
    codes, s, bits = _quant.quantize(x.cpu(), g, fmt, gs=gs)
    want = _quant.dequantize(codes, s, bits, g, fmt, gs).view(out.shape)
    got = out.cpu()
    nan = got.isnan()
    assert torch.equal(nan, want.isnan()) and torch.equal(got[~nan], want[~nan])


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("fmt,g", [("fp8_e4m3", 128), ("uint4", 64), ("nvfp4", 16)])
@pytest.mark.parametrize("gpu_index", [False, True])
def test_gather_then_quantize_into_cache_blocks(dev, fmt, g, gpu_index):
    # Pages gathered from a KV cache, quantized, written to other blocks of a
    # quantized cache: x_index and index, with host or GPU indices.
    if gpu_index and dev == "cpu":
        pytest.skip("GPU indices need GPU tensors")
    blocks, D = 32, 2 * g
    kv = data((blocks, 4), D, torch.bfloat16, dev, seed=5)          # [blocks, tokens, D]
    q, scale, zero = outputs(fmt, (blocks, 4), D, g, dev)
    gs = 0.02 if fmt == "nvfp4" else 1.0
    for e in engines():
        for k in range(6):
            src = torch.randperm(blocks)[:8]
            dst = torch.randperm(blocks)[:8]
            si, di = (src.to(dev), dst.to(dev)) if gpu_index else (src.numpy(), dst.tolist())
            e.quantize(kv, q, scale, zero, fmt=fmt, group=g, global_scale=gs, x_index=si, index=di)
            if dev == "cuda":
                torch.cuda.synchronize()
            check(fmt, g, kv[src], rows(q, dst), rows(scale, dst), None if zero is None else zero[dst], gs)
            back = torch.zeros(blocks, 4, D, dtype=torch.float32, device=dev)
            e.dequantize(q, scale, back, zero, fmt=fmt, group=g, global_scale=gs, x_index=si, index=di)
            if dev == "cuda":
                torch.cuda.synchronize()
            codes, s, bits = _quant.quantize(kv[src].cpu(), g, fmt, gs=gs)
            want = _quant.dequantize(codes, s, bits, g, fmt, gs).view(8, 4, D)
            got = back[src].cpu()
            nan = got.isnan()
            assert torch.equal(nan, want.isnan()) and torch.equal(got[~nan], want[~nan])


def test_out_of_range_index_raises():
    x = torch.randn(4, 4, 128)
    q, scale, _ = outputs("fp8_e4m3", (4, 4), 128, 128, "cpu")
    with pytest.raises(ValueError):
        ce.quantize(x, q, scale, fmt="fp8_e4m3", group=128, index=[0, 1, 9, 2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_out_of_range_gpu_index_raises_at_synchronize():
    x = torch.randn(4, 4, 128, device="cuda")
    q, scale, _ = outputs("fp8_e4m3", (4, 4), 128, 128, "cuda")
    e = ce.Engine(prior_only=True)
    e.quantize(x, q, scale, fmt="fp8_e4m3", group=128, index=torch.tensor([0, 1, 9, 2], device="cuda"))
    with pytest.raises(ValueError, match="out of range"):
        e.synchronize()
    e.quantize(x, q, scale, fmt="fp8_e4m3", group=128, index=torch.tensor([0, 1, 3, 2], device="cuda"))
    e.synchronize()  # cleared
