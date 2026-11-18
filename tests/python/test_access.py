"""Where and when memory is touched. torch's own copy touches host memory on the
CPU when it converts, gathers or copies pageable memory: GPU work on that host
memory must be done first (fuzzer seeds 90 and 145 read stale host bytes about
one run in twenty). A queue flushed because work arrives on another stream
orders the two streams."""

import os
import sys

import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fuzz"))


@pytest.mark.parametrize("seed", [90, 145])
def test_natives_after_gpu_work_on_host_memory(seed):
    import fuzz

    init, ops = fuzz.make_program(seed)
    plain = ops if seed == 145 else [op for op in ops if not op[1].indexed() and not op[2].indexed()]
    want = fuzz.reference(init, plain)
    kinds = [fuzz.kind_of(b) for b in init]
    for r in range(3):
        eng = ce.Engine()
        for it in range(40):  # exploration runs every strategy, native among them, in every order
            bs = [fuzz.copy_like(b, k) for b, k in zip(init, kinds)]
            if seed == 145:
                fuzz.run_explicit(eng, bs, plain, False)
            else:
                with eng.scope():
                    for _, dv, sv in plain:
                        dv.plain(bs[dv.base]).copy_(sv.plain(bs[sv.base]), non_blocking=True)
            torch.cuda.synchronize()
            assert all(fuzz.same(g, w) for g, w in zip(bs, want)), (r, it, eng.stats()["last"]["decisions"])


def test_a_flush_for_another_stream_orders_both():
    # The flush puts copyengine's copies on `a` and torch's on `b`: afterwards
    # each stream waits for the other (b waiting only for a's work before the
    # flush let b read `mid` while a still wrote it).
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    src = torch.randn(1 << 20, device=DEV)
    mid = torch.zeros_like(src)
    out = torch.zeros_like(src)
    e = ce.Engine()
    torch.cuda.synchronize()
    with e.scope():
        with torch.cuda.stream(a):
            torch.cuda._sleep(50_000_000)  # stream a is busy for a while
            mid.copy_(src)                 # queued on a
        with torch.cuda.stream(b):
            out.copy_(mid)                 # arrives on b: the queue is flushed, b must wait for a's copy
    torch.cuda.synchronize()
    assert torch.equal(out, src), str(e.stats()["last"])


@pytest.mark.parametrize("intercept", ["mode", "shim"])
@pytest.mark.parametrize("lazy", [False, True])
def test_natives_inside_a_scope_are_not_intercepted_again(intercept, lazy):
    # torch's own copy run by the engine (a native) inside an intercepting scope
    # must be done, not intercepted and queued again (fuzzer seed 1963: a later
    # copy then ran before it).
    if intercept == "shim" and ce._shim.lib is None:
        pytest.skip(ce._shim.error)
    e = ce.Engine(_force="native", _intercept=intercept)
    src = torch.randn(64, 32, device=DEV)
    want = src[:, ::2].clone()
    dst = torch.zeros(64, 16, device=DEV)
    with e.scope():
        if lazy:
            e.copy_(ce.lazy(dst), ce.lazy(src)[:, ::2])
        else:
            e.copy_(dst, src[:, ::2])
        e.barrier()                 # flushed while the scope intercepts: the native runs now
        assert e._e.queued == 0     # nothing came back into the queue
        src.fill_(-1.0)             # later work on the source
    torch.cuda.synchronize()
    assert torch.equal(dst, want)
