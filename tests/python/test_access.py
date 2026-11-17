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
            out.copy_(mid)                 # arrives on b: the queue is flushed, b must wait for a
    torch.cuda.synchronize()
    assert torch.equal(out, src)
