"""Bounds and gates found by the 2026-11-17 review: index lengths the kernels
read, opaque copies grouped with run-described ones, dtype gates on baselines,
zero-dimensional host indices."""

import numpy as np
import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"


def test_quantize_op_checks_index_lengths():
    x = torch.randn(16, 128, device=DEV)
    q = torch.zeros(16, 128, dtype=torch.int8, device=DEV)
    s = torch.zeros(16, 1, device=DEV)
    xi = torch.arange(8, device=DEV)
    ci = torch.arange(4, device=DEV)  # fewer than x_index: the kernel would read past it
    op = torch.ops.copyengine.quantize_
    with pytest.raises((RuntimeError, ValueError), match="indices, not rows"):
        op(x, q, s, None, 2, 128, 1, 1e-12, 1.0, False, xi, ci, 0)  # fmt 2: int8


def test_opaque_and_run_copies_in_one_scope():
    # A gather with GPU indices and a segments copy of the same sides: they used to share a
    # group whose forms were empty, and the flush failed for copies that work alone.
    src = torch.randint(0, 255, (64, 32), dtype=torch.uint8, device=DEV)
    out = torch.zeros(8, 32, dtype=torch.uint8, device=DEV)
    a = torch.randint(0, 255, (4096,), dtype=torch.uint8, device=DEV)
    b = torch.zeros(4096, dtype=torch.uint8, device=DEV)
    idx = torch.randperm(64, device=DEV)[:8]
    e = ce.Engine()
    with e.scope():
        e.gather(src, idx, 0, out=out)
        e.segments(a, b, np.array([0, 1024]), np.array([2048, 0]), np.array([512, 512]))
    torch.cuda.synchronize()
    assert torch.equal(out, src[idx])
    assert torch.equal(b[2048:2560], a[0:512]) and torch.equal(b[0:512], a[1024:1536])


def test_copy_many_never_takes_torch_bits_for_other_dtype_pairs():
    src = [torch.tensor([float("nan"), 3.7e9, -1.5, 2.5, 1e30], device=DEV) for _ in range(4)]
    e = ce.Engine()
    for k in range(60):
        dst = [torch.zeros(5, dtype=torch.int32, device=DEV) for _ in range(4)]
        e.copy_many(dst, src)
    torch.cuda.synchronize()
    sites = e.stats()["copy_many"]
    assert all(s["copyengine"] is not False for s in sites), sites  # never decided for torch's copies


def test_zero_dimensional_host_index_in_a_step_plan():
    src = torch.randn(16, 8, device=DEV)
    out = torch.zeros(1, 8, device=DEV)
    e = ce.Engine()

    def step(out, src, i):
        e.copy_(ce.lazy(out), ce.lazy(src).index_select(0, i))

    p = e.plan(step)
    for k in range(40):
        i = np.array(k % 16)  # 0-d
        p(out, src, i)
        torch.cuda.synchronize()
        assert torch.equal(out[0], src[k % 16]), k


def test_frozen_plan_owns_its_index_tensors():
    import gc
    src = torch.randn(64, 32, device=DEV)
    dst = torch.zeros(8, 32, device=DEV)
    e = ce.Engine()
    for _ in range(30):
        e.copy_(dst, ce.lazy(src).index_select(0, torch.arange(8, device=DEV) * 3))  # a temporary index tensor
    torch.cuda.synchronize()
    fr = e.freeze()
    gc.collect()
    junk = [torch.full((8,), 10**6, dtype=torch.int64, device=DEV) for _ in range(64)]  # would reuse a freed block
    dst.zero_()
    fr.run()
    torch.cuda.synchronize()
    assert torch.equal(dst, src[torch.arange(8, device=DEV) * 3])
    del junk


def test_step_plan_notices_new_memory_behind_the_same_tensor():
    src = torch.randn(64, 32, device=DEV)
    out = torch.zeros(8, 32, device=DEV)
    p = ce.Engine().plan(lambda out, src, ids: ce.gather(src, ids, 0, out=out))
    ids = torch.arange(8, device=DEV)
    for _ in range(300):
        p(out, src, ids)
    torch.cuda.synchronize()
    assert next(iter(p._layouts.values()))["traces"]
    out.data = torch.zeros(8, 32, device=DEV)  # the same Python object, other memory
    p(out, src, ids)
    torch.cuda.synchronize()
    assert torch.equal(out, src[ids])


def test_index_copies_refuse_overlap_as_torch_does():
    # torch's index_copy_ refuses a source overlapping its destination; so does ce.scatter_
    # with GPU indices (one pass would read what it writes). Lazy copies take two passes instead.
    pool = torch.randn(32, 16, device=DEV)
    idx = torch.randperm(32, device=DEV)[:8]
    with pytest.raises(RuntimeError):
        pool.index_copy_(0, idx, pool[:8])
    with pytest.raises(ce.UnsafeCopy):
        ce.Engine().scatter_(pool, idx, pool[:8], 0)


def test_frozen_two_pass_copy_owns_its_temporary():
    import gc
    pool = torch.randn(32, 4, 16, device=DEV)
    d = torch.tensor([3, 7, 1, 9], device=DEV)
    s = torch.tensor([7, 2, 3, 1], device=DEV)  # intersects d: two passes through a temporary
    e = ce.Engine()
    for _ in range(30):
        e.copy_(ce.lazy(pool).index_select(0, d), ce.lazy(pool).index_select(0, s))
    torch.cuda.synchronize()
    fr = e.freeze()
    gc.collect()
    junk = [torch.full((4, 4, 16), 1e6, device=DEV) for _ in range(64)]  # would reuse a freed temporary
    start = torch.randn_like(pool)
    pool.copy_(start)
    want = start.clone()
    want[d] = want[s]
    fr.run()
    torch.cuda.synchronize()
    assert torch.equal(pool, want)
    del junk


def test_reshape_merging_nested_strides_needs_no_table():
    # A merge of dimensions whose strides nest is one strided dimension, also
    # when another dimension has an index table (no table of offsets is built).
    a = ce._core.Access.contiguous([6, 4, 5]).index_select(0, np.array([4, 1, 3], dtype=np.int64))
    r = a.reshape([3, 20])
    assert not r.select(0, 0).has_table
    want = np.arange(6 * 4 * 5).reshape(6, 4, 5)[[4, 1, 3]].reshape(3, 20)
    np.testing.assert_array_equal(r.offsets().reshape(3, 20), want)
    t = ce._core.Access.contiguous([6, 4, 5]).permute([0, 2, 1]).index_select(0, np.array([2, 0], dtype=np.int64))
    m = t.reshape([2, 20])  # strides 1 and 5: they do not nest, a table it is
    assert m.select(0, 0).has_table
    want = np.arange(120).reshape(6, 4, 5).transpose(0, 2, 1)[[2, 0]].reshape(2, 20)
    np.testing.assert_array_equal(m.offsets().reshape(2, 20), want)
