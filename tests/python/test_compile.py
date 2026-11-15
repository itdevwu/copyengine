"""The explicit API inside torch.compile regions: one graph node per call
(no graph breaks), the same results as eager; forms no op takes break the
graph explicitly and still give the same results."""

import pytest

import copyengine as ce
from copyengine import _quant

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"
ops = pytest.mark.skipif(not ce._compiled.available(), reason="the torch ops need torch 2.10+")


def breaks(fn, *args):
    torch._dynamo.reset()
    return torch._dynamo.explain(fn)(*args).graph_break_count


@ops
@pytest.mark.parametrize("engine", ["module", "engine"])
def test_copy_is_one_node(engine):
    e = ce.Engine() if engine == "engine" else ce
    src = torch.randn(64, 128, device=DEV)
    dst = torch.empty(64, 128, device=DEV, dtype=torch.float16)

    def f(x):
        y = x * 2
        e.copy_(dst, y.t().contiguous().t())
        return dst.float().sum()

    assert breaks(f, src) == 0
    torch._dynamo.reset()
    out = torch.compile(f)(src)
    assert torch.equal(dst, (src * 2).to(torch.float16))
    assert torch.allclose(out, (src * 2).to(torch.float16).float().sum())


@ops
def test_copy_many_and_lazy_gather_scatter():
    blocks, n = 32, 8
    kv = [torch.randn(blocks, 16, 64, device=DEV) for _ in range(4)]
    host = [torch.zeros(n, 16, 64).pin_memory() for _ in kv]
    back = [torch.zeros(blocks, 16, 64, device=DEV) for _ in kv]
    dsts = [torch.empty_like(t) for t in kv]

    def step(idx):
        ce.copy_many(dsts, kv)
        for l in range(len(kv)):
            ce.copy_(host[l], ce.lazy(kv[l]).index_select(0, idx))
            ce.copy_(ce.lazy(back[l]).index_select(0, idx), host[l])
        return idx.sum()

    idx = torch.randperm(blocks, device=DEV)[:n]
    assert breaks(step, idx) == 0
    torch._dynamo.reset()
    f = torch.compile(step)
    for k in range(3):
        idx = torch.randperm(blocks, device=DEV)[:n]
        f(idx)
        torch.cuda.synchronize()
        for l in range(len(kv)):
            assert torch.equal(dsts[l], kv[l])
            assert torch.equal(host[l], kv[l].index_select(0, idx).cpu())
            assert torch.equal(back[l].index_select(0, idx), kv[l].index_select(0, idx))


@ops
def test_lazy_gather_then_permute():
    kv = torch.randn(16, 8, 4, 32, device=DEV)  # [blocks, tok, heads, d]
    out = torch.empty(4, 4, 8, 32, device=DEV)   # [n, heads, tok, d]

    def f(idx):
        ce.copy_(out, ce.lazy(kv).index_select(0, idx).permute(0, 2, 1, 3))
        return idx

    idx = torch.tensor([3, 0, 9, 15], device=DEV)
    assert breaks(f, idx) == 0
    torch._dynamo.reset()
    torch.compile(f)(idx)
    torch.cuda.synchronize()
    assert torch.equal(out, kv.index_select(0, idx).permute(0, 2, 1, 3))


@ops
@pytest.mark.parametrize("fmt,g", [("fp8_e4m3", 128), ("nvfp4", 16), ("uint4", 64)])
def test_quantize_is_one_node(fmt, g):
    blocks, D = 16, 2 * g
    gs = 0.02 if fmt == "nvfp4" else 1.0
    kv = torch.randn(blocks, 4, D, device=DEV).bfloat16()
    packed = fmt in _quant.PACKED
    qd = {"fp8_e4m3": torch.float8_e4m3fn}.get(fmt, torch.uint8)
    sd = {"nvfp4": torch.float8_e4m3fn}.get(fmt, torch.float32)
    q = torch.zeros(blocks, 4, D // 2 if packed else D, device=DEV, dtype=qd)
    scale = torch.zeros(blocks, 4, D // g, device=DEV, dtype=sd)
    zero = torch.zeros(blocks, 4, D // g, device=DEV, dtype=torch.uint8) if fmt == "uint4" else None
    back = torch.zeros(blocks, 4, D, device=DEV)

    def f(pages, slots):
        ce.quantize(kv, q, scale, zero, fmt=fmt, group=g, global_scale=gs, x_index=pages, index=slots)
        ce.dequantize(q, scale, back, zero, fmt=fmt, group=g, global_scale=gs, x_index=pages, index=slots)
        return pages

    pages, slots = torch.randperm(blocks, device=DEV)[:8], torch.randperm(blocks, device=DEV)[:8]
    assert breaks(f, pages, slots) == 0
    torch._dynamo.reset()
    torch.compile(f)(pages, slots)
    torch.cuda.synchronize()
    codes, s, bits = _quant.quantize(kv[pages].cpu(), g, fmt, gs=gs)
    got = q.view(torch.uint8)[slots].reshape(-1).cpu()
    nan = (got & 0x7F) == 0x7F if fmt == "fp8_e4m3" else torch.zeros_like(got, dtype=torch.bool)
    assert torch.equal(got[~nan], codes[~nan])
    want = _quant.dequantize(codes, s, bits, g, fmt, gs).view(8, 4, D)
    b = back[pages].cpu()
    nb = b.isnan()
    assert torch.equal(nb, want.isnan()) and torch.equal(b[~nb], want[~nb])


def test_forms_without_an_op_break_the_graph_and_match():
    kv = torch.randn(16, 8, 32, device=DEV)
    out = torch.empty(4, 8, 32, device=DEV)

    def f(x):
        ce.copy_(out, ce.lazy(kv).index_select(0, [3, 1, 4, 1]).flip(1))  # host indices, then a flip
        return x + 1

    torch._dynamo.reset()
    torch.compile(f)(torch.zeros(1, device=DEV))
    torch.cuda.synchronize()
    assert torch.equal(out, kv[[3, 1, 4, 1]].flip(1))


@ops
def test_reduce_overhead_cuda_graphs():
    src = torch.randn(256, 512, device=DEV)
    dst = torch.empty(256, 512, device=DEV, dtype=torch.bfloat16)

    def f(x):
        ce.copy_(dst, x * 3)
        return dst

    torch._dynamo.reset()
    g = torch.compile(f, mode="reduce-overhead")
    for k in range(5):
        x = src + k
        r = g(x)
        torch.cuda.synchronize()
        assert torch.equal(r, (x * 3).to(torch.bfloat16))
