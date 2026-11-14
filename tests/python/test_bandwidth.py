"""Bandwidth-limited copies (scope(bandwidth=...)): paced on the GPU, results
identical to torch. Rate checks are loose: the GPU may be shared."""

import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

DEV = "cuda"
MB = 1 << 20


def timed(fn):
    """GPU seconds of fn() on the current stream."""
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 1e3


def check_rate(nbytes, seconds, bw):
    rate = nbytes / seconds
    assert rate <= bw * 1.25, f"{rate / 1e9:.2f} GB/s over a {bw / 1e9:.2f} GB/s limit"
    assert rate >= bw * 0.2, f"{rate / 1e9:.2f} GB/s far below a {bw / 1e9:.2f} GB/s limit"


def paced(e):
    return e.stats()["last"]["strategies"] == ["paced"]


@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_pinned_transfer_stays_under_the_limit(direction):
    n = 48 * MB
    host = torch.randint(0, 255, (n,), dtype=torch.uint8).pin_memory()
    dev = torch.randint(0, 255, (n,), dtype=torch.uint8, device=DEV)
    src, dst = (host, dev) if direction == "h2d" else (dev, host)
    e = ce.Engine()
    bw = 2e9

    def run():
        with e.scope(bandwidth=bw):
            dst.copy_(src)

    run()  # plans and sets up pacing
    check_rate(n, timed(run), bw)
    assert paced(e)
    torch.cuda.synchronize()
    assert torch.equal(dst.cpu(), src.cpu())


def test_device_copy_and_explicit_api():
    n = 64 * MB
    src = torch.randn(n // 4, device=DEV)
    dst = torch.empty_like(src)
    e = ce.Engine(bandwidth=10e9)
    e.copy_(dst, src)
    check_rate(n, timed(lambda: e.copy_(dst, src)), 10e9)
    assert paced(e)
    assert torch.equal(dst, src)


def test_fragmented_rows_and_conversions_match_torch():
    host = torch.randn(4096, 256).pin_memory()
    e = ce.Engine()
    out = torch.empty(4096, 16, device=DEV)  # 64-byte rows: zero-copy kernel, not DMA
    with e.scope(bandwidth=1e9):
        out.copy_(host[:, 8:24])
    torch.cuda.synchronize()
    assert torch.equal(out, host[:, 8:24].to(DEV))
    half = torch.empty(4096, 256, device=DEV, dtype=torch.float16)
    with e.scope(bandwidth=4e9):
        half.copy_(host)
    torch.cuda.synchronize()
    assert torch.equal(half, host.to(DEV).half())
    back = torch.empty(4096, 256, dtype=torch.bfloat16).pin_memory()
    with e.scope(bandwidth=4e9):
        back.copy_(half)
    torch.cuda.synchronize()
    assert torch.equal(back, half.cpu().bfloat16())


def test_pageable_host_memory():
    host = torch.randn(1 << 20)
    dev = torch.empty_like(host, device=DEV)
    e = ce.Engine()
    with e.scope(bandwidth=2e9):
        dev.copy_(host)
    torch.cuda.synchronize()
    assert torch.equal(dev.cpu(), host)
    back = torch.empty_like(host)
    with e.scope(bandwidth=2e9):
        back.copy_(dev)
    assert torch.equal(back, host)


def test_limit_applies_to_its_scope_only():
    a = torch.randn(1 << 20, device=DEV)
    b = torch.empty_like(a)
    h = torch.empty(1 << 20).pin_memory()
    e = ce.Engine()
    with e.scope():
        b.copy_(a)
        with e.scope(bandwidth=1e9):
            h.copy_(a)
    torch.cuda.synchronize()
    assert sorted(e.stats()["last"]["strategies"]) != ["paced", "paced"]
    assert "paced" in e.stats()["last"]["strategies"]
    assert torch.equal(b, a) and torch.equal(h, a.cpu())
    with e.scope():
        b.copy_(a)
    assert "paced" not in e.stats()["last"]["strategies"]


def test_async_stream_scope_never_waits_and_is_paced():
    n = 32 * MB
    host = torch.randint(0, 255, (n,), dtype=torch.uint8).pin_memory()
    dev = torch.empty(n, dtype=torch.uint8, device=DEV)
    e = ce.Engine()
    bw = 2e9
    with e.scope(stream="copy", bandwidth=bw) as h:
        dev.copy_(host)
    h.synchronize()
    st = e.copy_stream()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(st)
    with e.scope(stream="copy", bandwidth=bw) as h:
        dev.copy_(host)
    b.record(st)
    assert not h.done()  # returned while the paced transfer runs
    b.synchronize()
    check_rate(n, a.elapsed_time(b) / 1e3, bw)
    assert torch.equal(dev.cpu(), host)


def test_paced_copy_in_a_cuda_graph():
    n = 16 * MB
    host = torch.randint(0, 255, (n,), dtype=torch.uint8).pin_memory()
    dev = torch.empty(n, dtype=torch.uint8, device=DEV)
    e = ce.Engine()
    bw = 2e9
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with e.scope(bandwidth=bw):  # first use sets up pacing outside the capture
            dev.copy_(host)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with e.scope(bandwidth=bw):
            dev.copy_(host)
    dev.zero_()
    host.copy_(torch.randint(0, 255, (n,), dtype=torch.uint8))
    check_rate(n, timed(g.replay), bw)
    assert torch.equal(dev.cpu(), host)


def test_invalid_limits():
    with pytest.raises(ValueError):
        ce.Engine().scope(bandwidth=-1).__enter__()
    with pytest.raises(ValueError):
        ce.Engine(bandwidth=float("nan"))
