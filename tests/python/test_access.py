"""Where and when memory is touched. torch's own copy touches host memory on the
CPU when it converts, gathers or copies pageable memory: GPU work on that host
memory must be done first (fuzzer seeds 90 and 145 read stale host bytes about
one run in twenty). A queue flushed because work arrives on another stream
orders the two streams."""

import os
import sys

import numpy as np
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


@pytest.mark.parametrize("force", [None, "fused"])
def test_a_copy_reading_gpu_indices_runs_before_a_later_copy_overwrites_them(force):
    # The kernel reads its index table when it runs: that read orders like any other
    # (before: a later copy into the index tensor could run first, in an earlier wave).
    e = ce.Engine(_force=force) if force else ce.Engine()
    src = torch.arange(64.0, device=DEV).view(64, 1).expand(64, 8).contiguous()
    for _ in range(5):
        ids = torch.tensor([3, 5, 7, 9], device=DEV)
        out = torch.zeros(4, 8, device=DEV)
        scratch, other = torch.zeros(1024, device=DEV), torch.ones(1024, device=DEV)
        with e.scope():
            e.copy_(scratch, other)                                   # unrelated: an earlier group
            e.gather(src, ids, 0, out=out)                            # reads ids on the GPU
            e.copy_(ids, torch.tensor([60, 61, 62, 63], device=DEV))  # then overwrites them
        torch.cuda.synchronize()
        assert torch.equal(out[:, 0], torch.tensor([3.0, 5, 7, 9], device=DEV)), e.stats()["last"]


def test_a_plan_reused_where_a_table_is_now_written_orders_it():
    # Two flushes with the same fingerprints (index tables' addresses are not in them):
    # in the second the later copy writes the very table the gather reads.
    e = ce.Engine()
    src = torch.arange(64.0, device=DEV).view(64, 1).expand(64, 8).contiguous()
    a, b = torch.tensor([1, 2, 3, 4], device=DEV), torch.tensor([10, 11, 12, 13], device=DEV)
    new = torch.tensor([50, 51, 52, 53], device=DEV)
    out = torch.zeros(4, 8, device=DEV)
    scratch, other = torch.zeros(1024, dtype=torch.int64, device=DEV), torch.ones(1024, dtype=torch.int64, device=DEV)
    for ids, want in ((a, [1.0, 2, 3, 4]), (b, [10.0, 11, 12, 13])):
        with e.scope():
            e.copy_(scratch, other)  # an earlier group the later copy joins
            e.gather(src, ids, 0, out=out)
            e.copy_(b, new)  # the first time another tensor, the second time the table read
        torch.cuda.synchronize()
        assert torch.equal(out[:, 0], torch.tensor(want, device=DEV)), (want, e.stats()["last"])
        b.copy_(torch.tensor([10, 11, 12, 13], device=DEV))


def test_a_copy_writing_its_own_index_table_is_refused():
    e = ce.Engine()
    buf = torch.zeros(16, dtype=torch.int64, device=DEV)
    ids = buf[:4]
    ids.copy_(torch.tensor([4, 5, 6, 7], device=DEV))
    with pytest.raises(ce.UnsafeCopy):
        e.scatter_(buf, ids, torch.ones(4, dtype=torch.int64, device=DEV))


def test_a_quantization_writing_its_own_index_table_is_refused():
    # The scales written over the GPU index table the kernel reads (codex review, dev10).
    e = ce.Engine()
    ids = torch.tensor([3, 5], dtype=torch.int64, device=DEV)  # 16 bytes = 2 rows x 2 float32 scales
    x = torch.randn(8, 256, device=DEV, dtype=torch.bfloat16)
    q = torch.empty(2, 256, dtype=torch.float8_e4m3fn, device=DEV)
    with pytest.raises(ce.UnsafeCopy):
        e.quantize(x, q, ids.view(torch.float32).view(2, 2), fmt="fp8_e4m3", group=128, x_index=ids)


def test_a_plan_keeps_which_quantization_reads_a_gpu_table():
    # Flush 1: A reads GPU ids, B host ids; flush 2 the other way round, a copy into ids
    # between them both times, the same ids tensor: the plans must not be shared, B must
    # read the ids the copy wrote (codex review, dev10; fingerprints say which op reads a table).
    e = ce.Engine()
    x = torch.randn(64, 256, device=DEV, dtype=torch.bfloat16)
    qa, qb = (torch.zeros(4, 256, dtype=torch.float8_e4m3fn, device=DEV) for _ in range(2))
    sa, sb = (torch.zeros(4, 2, device=DEV) for _ in range(2))
    host = np.array([0, 1, 2, 3])
    first, then = [9, 10, 11, 12], [40, 41, 42, 43]
    ref = ce.Engine()
    want_q = torch.empty_like(qa)
    want_s = torch.empty_like(sa)
    ref.quantize(x, want_q, want_s, fmt="fp8_e4m3", group=128, x_index=np.array(then))
    torch.cuda.synchronize()
    ids = torch.zeros(4, dtype=torch.int64, device=DEV)
    for k in range(4):
        ids.copy_(torch.tensor(first))
        a_idx, b_idx = (ids, host) if k % 2 == 0 else (host, ids)
        with e.scope():
            e.quantize(x, qa, sa, fmt="fp8_e4m3", group=128, x_index=a_idx)
            e.copy_(ids, torch.tensor(then, device=DEV))
            e.quantize(x, qb, sb, fmt="fp8_e4m3", group=128, x_index=b_idx)
        torch.cuda.synchronize()
        if k % 2:
            assert torch.equal(qb.view(torch.uint8), want_q.view(torch.uint8)), (k, e.stats()["last"])
            assert torch.equal(sb, want_s), k


def test_disabled_engines_run_indexed_copies_without_a_torch_equivalent():
    # disable=True runs every copy as its baseline: one with indices read on the GPU and a
    # function has no torch copy, and passthrough (runs) cannot run it: its own strategy does.
    src = torch.arange(64 * 8, dtype=torch.float32, device=DEV).view(64, 8)
    ids = torch.tensor([3, 5, 7], device=DEV)
    out = torch.zeros(3, 8, device=DEV)
    ce.Engine(disable=True).copy_(ce.lazy(out), ce.lazy(src)[ids], fn=ce.Elementwise(scale=2.0))
    torch.cuda.synchronize()
    assert torch.equal(out, src[ids] * 2)
