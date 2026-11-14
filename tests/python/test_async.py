"""scope(stream=...): copies run on another stream, the host never waits,
Pending orders them against other streams, buffers stay alive."""

import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"
BUSY = 1_000_000_000  # GPU clocks: the copies queue up behind this


def impls():
    out = [pytest.param("mode", id="mode")]
    out.append(pytest.param("shim", id="shim", marks=pytest.mark.skipif(ce._shim.lib is None, reason=ce._shim.error)))
    return out


@pytest.mark.parametrize("intercept", impls())
def test_copies_run_on_the_copy_stream_without_blocking_the_host(intercept):
    e = ce.Engine(_intercept=intercept)
    src = torch.randn(64, 256).pin_memory()
    dst = torch.zeros(64, 256, device=DEV)
    cur = torch.cuda.current_stream()
    # Warmup may wait (device setup, exploration); the steady state must not.
    for _ in range(60):
        with e.scope(stream="copy"):
            for r in range(64):
                dst[r].copy_(src[r])
    torch.cuda.synchronize()
    torch.cuda._sleep(BUSY)  # the current stream is busy: the copy stream waits for it
    with e.scope(stream="copy") as h:
        for r in range(64):
            dst[r].copy_(src[r])  # blocking copy_ semantics would wait for the GPU here
        assert torch.cuda.current_stream() == e.copy_stream()
    assert torch.cuda.current_stream() == cur
    assert not h.done(), "the host waited for the copies"
    h.synchronize()
    torch.testing.assert_close(dst.cpu(), src, rtol=0, atol=0)
    assert h.done() and h.stream == e.copy_stream()


def test_wait_orders_the_current_stream_after_the_copies():
    e = ce.Engine()
    x = torch.randn(1 << 20, device=DEV)
    y = torch.zeros_like(x)
    with e.scope(stream="copy") as h:
        torch.cuda._sleep(BUSY)  # on the copy stream: the copy is late
        e.copy_(y, x)
    h.wait()
    total = y.sum()  # on the current stream, after the copy
    torch.testing.assert_close(total, x.sum())


def test_the_copy_stream_sees_work_queued_before_the_scope():
    e = ce.Engine()
    x = torch.zeros(1 << 20, device=DEV)
    y = torch.zeros_like(x)
    torch.cuda._sleep(BUSY)
    x.fill_(3.0)  # still queued on the current stream when the scope starts
    with e.scope(stream="copy") as h:
        e.copy_(y, x)
    h.synchronize()
    assert bool((y == 3.0).all())


def test_pinned_sources_stay_alive_until_done():
    e = ce.Engine(prior_only=True)
    n = 1 << 20
    src = torch.arange(n, dtype=torch.float32).pin_memory()
    want, p = src.clone(), src.data_ptr()
    dst = torch.zeros(n, device=DEV)
    with e.scope(stream="copy") as h:
        torch.cuda._sleep(2 * BUSY)
        dst.copy_(src)
    del src
    assert not h.done()
    junk = torch.empty(n, pin_memory=True)
    assert junk.data_ptr() != p, "pinned memory of an in-flight copy was reused"
    h.synchronize()
    torch.testing.assert_close(dst.cpu(), want, rtol=0, atol=0)


def test_user_stream_and_nesting():
    e = ce.Engine()
    s = torch.cuda.Stream()
    a = torch.randn(4096, device=DEV)
    b, c = torch.zeros_like(a), torch.zeros_like(a)
    with e.scope():
        with e.scope(stream=s) as h:  # inside an outer scope: still issued at its own exit
            e.copy_(b, a)
        h.wait()
        e.copy_(c, b)  # ordered after the inner copies through h.wait()
    torch.cuda.synchronize()
    torch.testing.assert_close(c, a, rtol=0, atol=0)
    assert h.stream is s


def test_handle_before_exit_is_an_error_and_bad_streams_are_rejected():
    e = ce.Engine()
    with pytest.raises(ValueError):
        with e.scope(stream="elsewhere"):
            pass
    with e.scope(stream="copy") as h:
        with pytest.raises(RuntimeError, match="still open"):
            h.wait()


def test_inside_a_cuda_graph():
    e = ce.Engine()
    e.copy_stream()  # created before capturing
    x = torch.randn(64, 128, device=DEV)
    y = torch.zeros(128, 64, device=DEV)
    z = torch.zeros(128, 64, device=DEV)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            with e.scope(stream="copy") as h:  # fork
                e.copy_(y, x.t())
            h.wait()  # join
            z.copy_(y * 2)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(z, x.t() * 2, rtol=0, atol=0)
