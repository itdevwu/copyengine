"""CUDA Graph capture: copies inside torch.cuda.graph are recorded and replay
correctly; nothing in a capture may allocate, synchronize or time."""

import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


def capture(fn):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    return g


def replayed(g, src, dst, want):
    """Replay with new source data: the destination follows it."""
    for k in range(3):
        src.copy_(torch.randn_like(src.float()).to(src.dtype) if src.is_floating_point() else src)
        dst.zero_()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(dst, want(), rtol=0, atol=0)


@pytest.mark.parametrize("warm", [False, True])
def test_explicit_device_copy(warm):
    x = torch.randn(64, 128, device=DEV)
    y = torch.zeros(128, 64, device=DEV)
    e = ce.Engine()
    for _ in range(80 if warm else 0):
        e.copy_(y, x.t())
    g = capture(lambda: e.copy_(y, x.t()))
    replayed(g, x, y, lambda: x.t())


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("intercept", ["mode", "shim"])
def test_intercepted_scope(warm, intercept):
    if intercept == "shim" and ce._shim.lib is None:
        pytest.skip(ce._shim.error)
    L, B, N, S = 4, 16, 16, 8
    k = torch.rand(B, N, 1, device=DEV)
    scale = torch.zeros(L, B, N, S, device=DEV)
    e = ce.Engine(_intercept=intercept)

    def body():
        with e.scope():
            for l in range(L):
                for b in range(B):
                    scale[l, b].copy_(k[b].expand(N, S))

    for _ in range(80 if warm else 0):
        body()
    g = capture(body)
    replayed(g, k, scale, lambda: k.expand(B, N, S).unsqueeze(0).expand(L, B, N, S))


@pytest.mark.parametrize("warm", [False, True])
def test_h2d_from_pinned(warm):
    h = torch.randn(256, 64).pin_memory()
    d = torch.zeros(256, 64, device=DEV)
    e = ce.Engine()
    for _ in range(80 if warm else 0):
        e.copy_(d, h)
    g = capture(lambda: e.copy_(d, h))
    for _ in range(3):
        h.copy_(torch.randn_like(h))
        d.zero_()
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(d.cpu(), h, rtol=0, atol=0)


@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty")  # the refused capture records nothing
def test_copy_without_torch_equivalent_needs_a_run_first():
    cache = torch.randn(512, 4, 32, device=DEV)
    ids = torch.randperm(512)[:200].tolist()
    out = torch.zeros(200, 32, 4, device=DEV)
    e = ce.Engine()
    src = lambda: ce.lazy(cache).index_select(0, ids).permute(0, 2, 1)
    with pytest.raises(ce.Unsupported, match="run it once before capturing"):
        capture(lambda: e.copy_(out, src()))
    for _ in range(40):  # explored and locked: its tables are resident now
        e.copy_(out, src())
    g = capture(lambda: e.copy_(out, src()))
    idx = torch.tensor(ids, device=DEV)
    replayed(g, cache, out, lambda: cache[idx].permute(0, 2, 1))


def test_frozen_plan_replays_inside_a_graph():
    L, B, N = 8, 16, 64
    slot = torch.randn(L, B, N, device=DEV)
    cache = torch.zeros(L, 40, N, device=DEV)
    ids = torch.randperm(40)[:B].tolist()
    e = ce.Engine()

    def body():
        with e.scope():
            for l in range(L):
                for i, b in enumerate(ids):
                    e.copy_(cache[l, b], slot[l, i])

    for _ in range(60):
        body()
    plan = e.freeze()
    g = capture(plan.run)
    idx = torch.tensor(ids, device=DEV)
    for _ in range(3):
        slot.copy_(torch.randn_like(slot))
        cache.zero_()
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(cache[:, idx], slot, rtol=0, atol=0)


def test_capture_leaves_the_engine_usable():
    x = torch.randn(1024, device=DEV)
    y = torch.zeros(1024, device=DEV)
    e = ce.Engine()
    capture(lambda: e.copy_(y, x))
    for _ in range(5):
        e.copy_(y, x * 2)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x * 2, rtol=0, atol=0)
