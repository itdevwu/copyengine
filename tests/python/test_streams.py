"""Copies on one stream never wait for an engine's copies on another stream:
copyengine adds no ordering between streams beyond the caller's own."""

import time

import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"


def _independent(e, copy_on, warm=30):
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    for s in (a, b):  # tuned and locked on both streams
        with torch.cuda.stream(s):
            for _ in range(warm):
                copy_on()
    torch.cuda.synchronize()
    with torch.cuda.stream(b):
        torch.cuda._sleep(1_500_000_000)  # b is busy for a long time
        copy_on()
    with torch.cuda.stream(a):
        copy_on()
    t = time.perf_counter()
    a.synchronize()
    waited = time.perf_counter() - t
    busy = not b.query()
    torch.cuda.synchronize()
    assert waited < 0.05, f"stream a waited {waited * 1e3:.0f} ms for stream b's work"
    assert busy, "the busy stream finished too early for the check"


@pytest.mark.parametrize("max_sms", [0, 4])
def test_kernel_copies_on_two_streams(max_sms):
    src, dst = torch.randn(512, 256, device=DEV), torch.empty(256, 512, device=DEV)
    e = ce.Engine(max_sms=max_sms, prior_only=True)  # device transposes: a copy kernel
    _independent(e, lambda: e.copy_(dst, src.t()))


def test_staged_copies_on_two_streams():
    # Host-to-device copies may share one copy engine in hardware (then torch's
    # own copies on two streams wait for each other too): ce must not add to it.
    src = torch.randn(256, 512).pin_memory()
    dst, d2 = torch.empty(512, 256, device=DEV), torch.empty(256, 512, device=DEV)
    e = ce.Engine(prior_only=True, allow_async=True)  # pinned strided H2D: device staging

    def waited(copy_on):
        a, b = torch.cuda.Stream(), torch.cuda.Stream()
        for s in (a, b):
            with torch.cuda.stream(s):
                for _ in range(10):
                    copy_on()
        torch.cuda.synchronize()
        with torch.cuda.stream(b):
            torch.cuda._sleep(1_500_000_000)
            copy_on()
        with torch.cuda.stream(a):
            copy_on()
        t = time.perf_counter()
        a.synchronize()
        w = time.perf_counter() - t
        torch.cuda.synchronize()
        return w

    hw = waited(lambda: d2.copy_(src, non_blocking=True))
    ours = waited(lambda: e.copy_(dst, src.t()))
    assert ours < hw + 0.05, f"ce waited {ours * 1e3:.0f} ms, torch {hw * 1e3:.0f} ms"
