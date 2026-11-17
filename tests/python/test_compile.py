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


@pytest.fixture(autouse=True)
def no_baseline_measurement(monkeypatch):
    # Measuring the compiled baseline costs a compile per layout: only the tests about it measure.
    monkeypatch.setenv("CE_COMPILE_BASELINE", "0")


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
    assert torch.allclose(out, (src * 2).to(torch.float16).float().sum(), rtol=1e-4)  # summation order differs


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


def _graph_ops(fn, *args):
    torch._dynamo.reset()
    ex = torch._dynamo.explain(fn)(*args)
    return "\n".join(g.code for g in ex.graphs), ex.graph_break_count


@ops
def test_the_compiled_baseline_is_emitted_when_it_is_as_fast(monkeypatch):
    # The decision is measured once per layout; forced here both ways.
    monkeypatch.delenv("CE_COMPILE_BASELINE")
    src = torch.randn(64, 128, device=DEV)
    dst = torch.empty(128, 64, device=DEV, dtype=torch.float16)

    def f(x):
        ce.copy_(dst, (x * 2).t())
        return dst

    for verdict, has_op in (("torch", False), ("ce", True)):
        ce._compiled._DECISIONS.clear()
        monkeypatch.setattr(ce._compiled, "_measure", lambda *a, verdict=verdict, **k: (verdict, "forced"))
        code, breaks_ = _graph_ops(f, src)
        assert breaks_ == 0 and ("copyengine" in code) == has_op, (verdict, code)
        torch._dynamo.reset()
        torch.compile(f)(src)
        torch.cuda.synchronize()
        assert torch.equal(dst, (src * 2).t().to(torch.float16))
    ce._compiled._DECISIONS.clear()


@ops
def test_the_measurement_runs_and_is_kept(monkeypatch):
    monkeypatch.delenv("CE_COMPILE_BASELINE")
    ce._compiled._DECISIONS.clear()
    kv = torch.randn(32, 16, 64, device=DEV)
    out = torch.empty(8, 16, 64, device=DEV)

    def f(idx):
        ce.gather(kv, idx, 0, out=out)
        return idx

    idx = torch.randperm(32, device=DEV)[:8]
    torch._dynamo.reset()
    torch.compile(f)(idx)
    torch.cuda.synchronize()
    assert torch.equal(out, kv[idx])
    assert any(k[0] == "gather" for k in ce._compiled._DECISIONS)
    assert set(ce._compiled._DECISIONS.values()) <= {"ce", "torch"}
    why = [w for k, w in ce._compiled._REASONS.items() if k[0] == "gather"]
    assert why and "us" in why[0], why  # measured (not a baseline that failed to build)
    monkeypatch.setenv("CE_COMPILE_BASELINE", "0")
    code, _ = _graph_ops(f, idx)
    assert "copyengine" in code
    ce._compiled._DECISIONS.clear()


@ops
def test_warm_when_the_model_is_warm():
    # Module-level calls in a compiled model explore on its warmup runs and lock;
    # compiled_stats() shows the process-wide engine they use.
    blocks, n = 64, 16
    kv = torch.randn(blocks, 32, 128, device=DEV)
    host = torch.empty(n, 32, 128).pin_memory()

    def step(idx):
        ce.copy_(host, ce.lazy(kv).index_select(0, idx))
        return idx

    torch._dynamo.reset()
    g = torch.compile(step)
    for _ in range(60):
        idx = torch.randperm(blocks, device=DEV)[:n]
        g(idx)
    torch.cuda.synchronize()
    assert torch.equal(host, kv[idx].cpu())
    st = ce.compiled_stats()
    whys = [d["why"] for d in st["last"]["decisions"]]
    assert whys and all(w.startswith("locked") for w in whys), whys


@ops
def test_dynamic_shapes_keep_the_graph(monkeypatch):
    # Symbolic sizes never reach the compiled-baseline decision (a constant of
    # the graph): such calls take copyengine's op, in one graph.
    monkeypatch.delenv("CE_COMPILE_BASELINE")
    for n in (16, 24):
        x = torch.randn(n, 32, device=DEV)
        dst = torch.empty(n, 32, device=DEV, dtype=torch.float16)
        kv = torch.randn(n, 8, 64, device=DEV)
        idx = torch.arange(4, device=DEV)
        out = torch.empty(4, 8, 64, device=DEV)

        def f(x, dst, kv, idx, out):
            ce.copy_(dst, x * 2)
            ce.gather(kv, idx, 0, out=out)
            return dst

        torch._dynamo.reset()
        for t in (x, dst, kv):
            torch._dynamo.mark_dynamic(t, 0)
        ex = torch._dynamo.explain(f)(x, dst, kv, idx, out)
        assert ex.graph_count == 1 and ex.graph_break_count == 0, [str(r.reason)[:200] for r in ex.break_reasons]
        torch.compile(f, dynamic=True)(x, dst, kv, idx, out)
        torch.cuda.synchronize()
        assert torch.equal(dst, (x * 2).to(torch.float16)) and torch.equal(out, kv[idx])


@ops
def test_reshape_of_a_split_tensor_stays_in_the_graph():
    # v of a fused QKV projection: not contiguous (rows skip q and k), yet its
    # reshape is a view: no graph break.
    qkv = torch.randn(4, 16, 3 * 64, device=DEV)
    o = torch.empty(4, 2, 16, 32, device=DEV)

    def f(qkv, o):
        v = qkv[..., 128:]
        ce.copy_(o, ce.lazy(v).reshape(4, 16, 2, 32).permute(0, 2, 1, 3))
        return o

    assert breaks(f, qkv, o) == 0
    torch._dynamo.reset()
    torch.compile(f)(qkv, o)
    torch.cuda.synchronize()
    assert torch.equal(o, qkv[..., 128:].reshape(4, 16, 2, 32).permute(0, 2, 1, 3))


def test_block_copy_within_one_pool_matches_eager():
    pool = torch.randn(32, 4, 16, device="cuda")
    d = torch.tensor([3, 7, 1, 9], device="cuda")
    s = torch.tensor([7, 2, 3, 1], device="cuda")  # intersects d: torch gathers first

    def f(pool, d, s):
        ce.copy_(ce.lazy(pool).index_select(0, d), ce.lazy(pool).index_select(0, s).permute(0, 1, 2))
        return pool

    want = pool.clone()
    want[d] = want[s]
    got = torch.compile(f)(pool.clone(), d, s)
    torch.cuda.synchronize()
    assert torch.equal(got, want)
