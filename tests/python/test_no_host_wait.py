"""Hot paths never synchronize: once warmed up, each API returns while the GPU
is still busy with earlier work. A sync anywhere (an event or stream wait, a
blocking free, a device-to-host read) makes the host wait for the long kernel
queued first."""

import time

import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"
BUSY_S = 0.4  # the GPU stays busy at least this long


def _cycles():
    # Calibrate torch.cuda._sleep: cycles for BUSY_S on this GPU.
    torch.cuda.synchronize()
    t = time.perf_counter()
    torch.cuda._sleep(10_000_000)
    torch.cuda.synchronize()
    return 2 * int(10_000_000 * BUSY_S / max(time.perf_counter() - t, 1e-4))  # margin: clocks vary


CYCLES = None


def returns_while_busy(call, reps=20):
    global CYCLES
    if CYCLES is None:
        CYCLES = _cycles()
    for _ in range(reps):  # steady state first
        call()
    torch.cuda.synchronize()
    torch.cuda._sleep(CYCLES)
    t = time.perf_counter()
    for _ in range(reps):
        call()
    host = time.perf_counter() - t
    torch.cuda.synchronize()
    assert host < BUSY_S / 4, f"the host waited {host * 1e3:.0f} ms for the GPU"


def test_scope_of_copies():
    src = [torch.randn(64, 64, device=DEV) for _ in range(32)]
    dst = [torch.empty(64, 64, device=DEV) for _ in range(32)]
    e = ce.Engine()

    def call():
        with e.scope():
            for d, s in zip(dst, src):
                d.copy_(s)

    returns_while_busy(call)


def test_explicit_copies():
    src, dst = torch.randn(256, 512, device=DEV), torch.empty(512, 256, device=DEV)
    e = ce.Engine()
    returns_while_busy(lambda: e.copy_(dst, src.t()))


@pytest.mark.parametrize("fresh", [False, True])
def test_gather_with_device_indices(fresh):
    src, out = torch.randn(1024, 64, device=DEV), torch.empty(32, 64, device=DEV)
    pool = [torch.randperm(1024, device=DEV)[:32] for _ in range(16)]
    e = ce.Engine()
    k = [0]

    def call():
        ids = pool[k[0] % len(pool)] if fresh else pool[0]
        k[0] += 1
        e.gather(src, ids, 0, out=out)

    returns_while_busy(call, reps=40)


def test_step_plan_replay():
    caches = [torch.randn(256, 64, device=DEV) for _ in range(4)]
    pool = torch.empty(4, 16, 64, device=DEV)
    ids = [torch.randperm(256, device=DEV)[:16] for _ in range(8)]
    plan = ce.Engine().plan(lambda caches, pool, i: [ce.gather(c, i, 0, out=pool[l]) for l, c in enumerate(caches)])
    k = [0]

    def call():
        plan(caches, pool, ids[k[0] % len(ids)])
        k[0] += 1

    for _ in range(80):  # trace, settle, decide
        call()
    returns_while_busy(call)


def test_async_scope_from_pinned_memory():
    src = [torch.randn(128, 128).pin_memory() for _ in range(8)]
    dst = [torch.empty(128, 128, device=DEV) for _ in range(8)]
    e = ce.Engine()

    def call():
        with e.scope(stream="copy") as h:
            for d, s in zip(dst, src):
                d.copy_(s, non_blocking=True)
        h.wait()

    returns_while_busy(call)


def test_bandwidth_limited_scope():
    src, dst = torch.randn(1 << 18, device=DEV), torch.empty(1 << 18, device=DEV)
    e = ce.Engine()

    def call():
        with e.scope(bandwidth=50e9):
            dst.copy_(src)

    returns_while_busy(call, reps=5)


def test_frozen_replay():
    src, dst = torch.randn(256, 512, device=DEV), torch.empty(512, 256, device=DEV)
    e = ce.Engine()
    for _ in range(20):
        e.copy_(dst, src.t())
    plan = e.freeze()
    returns_while_busy(plan.run)


def test_non_blocking_copies_in_a_plain_scope():
    # Intercepted copy_ keeps the caller's non_blocking: no wait at scope exit.
    src = [torch.randn(128, 128).pin_memory() for _ in range(8)]
    dst = [torch.empty(128, 128, device=DEV) for _ in range(8)]
    e = ce.Engine()

    def call():
        with e.scope():
            for d, s in zip(dst, src):
                d.copy_(s, non_blocking=True)

    returns_while_busy(call)


def test_the_check_catches_a_wait():
    with pytest.raises(AssertionError, match="waited"):
        returns_while_busy(torch.cuda.synchronize, reps=1)
