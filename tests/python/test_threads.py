"""One engine shared by threads: each thread has its own session (queue,
scopes, stream, last flush); plans and tuning are shared."""

import threading

import numpy as np
import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"
N = 4


def _run(threads):
    errors = []

    def wrap(f, k):
        try:
            f(k)
        except BaseException as ex:  # surfaced in the main thread
            errors.append(ex)

    ts = [threading.Thread(target=wrap, args=(f, k)) for k, f in enumerate(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errors:
        raise errors[0]


def test_scopes_on_one_engine_from_many_threads():
    e = ce.Engine()
    srcs = [torch.randn(64, 256, device=DEV) for _ in range(N)]
    hosts = [torch.randn(64, 256).pin_memory() for _ in range(N)]

    def work(k):
        s = torch.cuda.Stream()
        dst = torch.zeros(64, 128, device=DEV)
        dh = torch.zeros(64, 256, device=DEV)
        with torch.cuda.stream(s):
            for it in range(60):
                with e.scope():
                    dst.copy_(srcs[k][:, ::2])
                    dh.copy_(hosts[k])
                s.synchronize()
                assert torch.equal(dst, srcs[k][:, ::2]) and torch.equal(dh.cpu(), hosts[k]), (k, it)

    _run([work] * N)


def test_step_plans_of_many_threads_on_one_engine():
    e = ce.Engine()
    caches = [torch.randn(64, 32, device=DEV) for _ in range(N)]

    def work(k):
        out = torch.zeros(8, 32, device=DEV)
        p = e.plan(lambda out, src, ids: e.gather(src, ids, 0, out=out))
        for it in range(150):
            ids = torch.randperm(64, device=DEV)[:8]
            p(out, caches[k], ids)
            torch.cuda.synchronize()
            assert torch.equal(out, caches[k][ids]), (k, it)

    _run([work] * N)


def test_async_scopes_from_many_threads():
    e = ce.Engine()
    srcs = [torch.randn(1 << 18, device=DEV) for _ in range(N)]

    def work(k):
        dst = torch.zeros_like(srcs[k])
        for it in range(30):
            with e.scope(stream="copy") as h:
                dst.copy_(srcs[k])
            h.synchronize()
            assert torch.equal(dst, srcs[k]), (k, it)

    _run([work] * N)


def test_each_thread_freezes_its_own_last_flush():
    # Explicit copies without tuning always flush through copyengine (an intercepted
    # copy may run as torch's in a sampled scope, and then there is nothing to freeze).
    e = ce.Engine(prior_only=True)
    srcs = [torch.randn(32, 64, device=DEV) for _ in range(N)]
    dsts = [torch.zeros(32, 32, device=DEV) for _ in range(N)]
    barrier = threading.Barrier(N, timeout=60)
    frozen = [None] * N

    def work(k):
        for _ in range(20):
            with e.scope():
                e.copy_(dsts[k], srcs[k][:, k::2] if k < 2 else srcs[k][:, :32])
        barrier.wait()  # every thread flushed: each freeze must take its own
        frozen[k] = e.freeze()

    _run([work] * N)
    torch.cuda.synchronize()
    for k in range(N):
        dsts[k].zero_()
        frozen[k].run()
    torch.cuda.synchronize()
    for k in range(N):
        want = srcs[k][:, k::2] if k < 2 else srcs[k][:, :32]
        assert torch.equal(dsts[k], want), k


def test_freeze_after_torch_ran_refuses_even_when_another_thread_flushed():
    # Freeze compares this thread's flushes, not the engine's: other threads'
    # flushes must not make an older flush of this thread look like its last.
    e = ce.Engine()
    src, dst = torch.randn(32, 64, device=DEV), torch.zeros(32, 32, device=DEV)
    ran_torch, other_flushed = threading.Event(), threading.Event()
    seen = []

    def a(_):
        with e.scope():
            dst.copy_(src[:, ::2])
        e._copy_many_torch([dst], [src[:, 1::2]])  # the last copies ran as torch's own
        ran_torch.set()
        other_flushed.wait(60)
        with pytest.raises(RuntimeError, match="ran as torch's own"):
            e.freeze()
        seen.append("a")

    def b(_):
        ran_torch.wait(60)
        x, y = torch.zeros(64, device=DEV), torch.ones(64, device=DEV)
        for _ in range(3):
            with e.scope():
                x.copy_(y)
        other_flushed.set()
        seen.append("b")

    _run([a, b])
    assert sorted(seen) == ["a", "b"]


def test_host_copies_from_many_threads_numpy():
    e = ce.Engine()

    def work(k):
        rng = np.random.default_rng(k)
        for it in range(100):
            a = rng.standard_normal(1000).astype(np.float32)
            b = np.zeros_like(a)
            with e.scope():
                e.copy_(b[::2], a[1::2])
            np.testing.assert_array_equal(b[::2], a[1::2])

    _run([work] * N)


def test_sessions_of_exited_threads_are_let_go():
    # Threads that come and go (a pool resized, one thread per request) leave
    # nothing behind: their last flush and buffers are released once done.
    e = ce.Engine()
    for _ in range(3):
        def work(k):
            d = torch.zeros(1 << 16, device=DEV)
            with e.scope():
                d.copy_(torch.ones(1 << 16).pin_memory(), non_blocking=True)
            e.copy_(torch.zeros(8, device=DEV), torch.ones(8, device=DEV))
        _run([work] * 8)
    torch.cuda.synchronize()
    e.copy_(torch.zeros(8, device=DEV), torch.ones(8, device=DEV))
    t = threading.Thread(target=lambda: e.copy_(torch.zeros(8, device=DEV), torch.ones(8, device=DEV)))
    t.start()
    t.join()
    torch.cuda.synchronize()  # its last copy done: a session is let go once nothing of it is in flight
    e.copy_(torch.zeros(8, device=DEV), torch.ones(8, device=DEV))  # this thread again: the others are gone
    assert e.stats()["sessions"] == 1, (e.stats()["sessions"], e.stats()["last"])
