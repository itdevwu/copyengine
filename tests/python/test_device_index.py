"""Index tensors that stay on the GPU: the statement kernel reads them, the
host never sees the values. Checked against torch."""

import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

DEV = "cuda"


def test_gather_matches_torch_without_syncs():
    src = torch.randn(64, 8, 16, device=DEV)
    e = ce.Engine()
    for k in range(6):
        idx = torch.randint(0, 64, (20,), device=DEV)
        with torch.cuda.device(0):
            torch.cuda.set_sync_debug_mode("error")  # a host copy of idx would raise
            try:
                out = e.gather(src, idx, 0)
            finally:
                torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
        assert torch.equal(out, src.index_select(0, idx)), k


def test_new_indices_reuse_the_plan():
    src = torch.randn(32, 64, device=DEV)
    out = torch.empty(16, 64, device=DEV)
    e = ce.Engine()
    for _ in range(4):
        e.gather(src, torch.randperm(32, device=DEV)[:16], 0, out=out)
    plans = e.stats()["plans"]
    for _ in range(8):
        idx = torch.randperm(32, device=DEV)[:16]
        e.gather(src, idx, 0, out=out)
        torch.cuda.synchronize()
        assert torch.equal(out, src[idx])
        assert e.stats()["last"]["plan_cache"] == "hit"
    assert e.stats()["plans"] == plans


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_scatter_and_casts(dtype):
    dst = torch.zeros(40, 24, device=DEV, dtype=torch.float16)
    src = torch.randn(10, 24, device=DEV)
    idx = torch.randperm(40, device=DEV)[:10].to(dtype)
    ce.Engine().scatter_(dst, idx, src, 0)
    ref = torch.zeros_like(dst).index_copy_(0, idx.long(), src.half())
    torch.cuda.synchronize()
    assert torch.equal(dst, ref)


def test_lazy_view_with_device_indices():
    cache = torch.randn(16, 4, 8, 32, device=DEV)  # blocks, heads, tokens, dim
    idx = torch.tensor([3, 1, 7, 7, 0], device=DEV)
    out = torch.empty(5, 8, 4, 32, device=DEV)
    ce.Engine().copy_(out, ce.lazy(cache).index_select(0, idx).permute(0, 2, 1, 3))
    torch.cuda.synchronize()
    assert torch.equal(out, cache.index_select(0, idx).permute(0, 2, 1, 3))


def test_device_index_into_inner_and_sliced_dims():
    src = torch.randn(6, 50, device=DEV)
    idx = torch.randint(0, 25, (9,), device=DEV)
    e = ce.Engine()
    out = torch.empty(6, 9, device=DEV)
    e.copy_(out, ce.lazy(src).slice(1, 0, 50, 2).index_select(1, idx))
    torch.cuda.synchronize()
    assert torch.equal(out, src[:, ::2].index_select(1, idx))
    out2 = torch.empty(6, 4, device=DEV)
    big = torch.randint(0, 25, (8,), device=DEV)  # slice the device table itself
    e.copy_(out2, ce.lazy(src).index_select(1, big).slice(1, 1, 8, 2))
    torch.cuda.synchronize()
    assert torch.equal(out2, src.index_select(1, big)[:, 1::2])


def test_out_of_range_index_raises_at_synchronize():
    src = torch.arange(10.0, device=DEV)
    out = torch.full((4,), -1.0, device=DEV)
    e = ce.Engine()
    e.gather(src, torch.tensor([1, 12, 3, -1], device=DEV), 0, out=out)
    with pytest.raises(ValueError, match="out of range"):
        e.synchronize()
    assert out.tolist() == [1.0, -1.0, 3.0, -1.0]  # the bad elements were skipped
    e.gather(src, torch.tensor([0, 1, 2, 3], device=DEV), 0, out=out)
    e.synchronize()  # the error word was cleared
    assert out.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_ops_needing_values_refuse():
    src = torch.randn(8, 8, device=DEV)
    x = ce.lazy(src).index_select(0, torch.tensor([1, 2], device=DEV))
    with pytest.raises(ce.NeedsMaterialize):
        x.select(0, 1)
    with pytest.raises(ce.NeedsMaterialize):
        x.index_select(0, torch.tensor([0], device=DEV))


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_explicit_calls_through_the_torch_op(dim):
    src = torch.randn(12, 10, 6, device=DEV)
    n = src.shape[dim]
    idx = torch.randint(0, n, (5,), device=DEV)
    e = ce.Engine()
    out = e.gather(src, idx, dim)  # allocated
    torch.cuda.synchronize()
    assert torch.equal(out, src.index_select(dim, idx))
    dst = torch.zeros_like(src)
    perm = torch.randperm(n, device=DEV)[:5]
    e.scatter_(dst, perm, out, dim)
    torch.cuda.synchronize()
    assert torch.equal(dst, torch.zeros_like(src).index_copy_(dim, perm, out))
    row = out.select(dim, 0).unsqueeze(dim)  # broadcast source: no torch index_copy_ equivalent
    e.scatter_(dst, perm, row, dim)
    torch.cuda.synchronize()
    assert torch.equal(dst.index_select(dim, perm), row.expand_as(out))


def test_empty_device_index_is_a_no_op():
    src = torch.randn(8, 4, device=DEV)
    i0 = torch.empty(0, dtype=torch.long, device=DEV)
    e = ce.Engine()
    assert e.gather(src, i0, 0).shape == (0, 4)
    e.scatter_(src.clone(), i0, torch.empty(0, 4, device=DEV), 0)
    e.copy_(torch.empty(0, 4, device=DEV), ce.lazy(src).index_select(0, i0))
    e.synchronize()


def test_gathers_into_disjoint_slices_are_one_launch():
    # Their writes are known exactly (no table on the destination): no false
    # dependencies, one wave, one group.
    caches = [torch.randn(128, 32, device=DEV) for _ in range(6)]
    pool = torch.empty(6, 8, 32, device=DEV)
    ids = torch.randperm(128, device=DEV)[:8]
    e = ce.Engine(prior_only=True)
    with e.scope():
        for l, c in enumerate(caches):
            e.gather(c, ids, 0, out=pool[l])
    last = e.stats()["last"]
    assert last["waves"] == 1 and last["groups"] == 1 and last["calls"] == 1, last
    torch.cuda.synchronize()
    assert torch.equal(pool, torch.stack([c[ids] for c in caches]))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_device_indices_with_pinned_host_side(dtype):
    # Pages of a GPU cache by GPU block ids into pinned host buffers (offload),
    # and back into scattered GPU blocks (load): the indices never leave the GPU.
    blocks, tok, hd, n = 32, 16, 64, 8
    kv = torch.randn(blocks, tok, hd, device=DEV).to(dtype)
    host = torch.zeros(n, tok, hd, dtype=dtype).pin_memory()
    back = torch.zeros(blocks, tok, hd, device=DEV, dtype=dtype)
    e = ce.Engine(tune_trials=1, tune_topk=4)
    seen = set()
    for k in range(10):
        idx = torch.randperm(blocks, device=DEV)[:n]
        host.zero_()
        e.copy_(host, ce.lazy(kv).index_select(0, idx))
        e.synchronize()
        seen.update(e.stats()["last"]["strategies"])
        assert torch.equal(host, kv.index_select(0, idx).cpu())
        back.zero_()
        e.copy_(ce.lazy(back).index_select(0, idx), host)
        e.synchronize()
        seen.update(e.stats()["last"]["strategies"])
        assert torch.equal(back.index_select(0, idx), kv.index_select(0, idx))
        assert int((back != 0).any(-1).any(-1).sum()) <= n  # nothing outside the selected blocks
        g = e.gather(kv, idx, 0, out=torch.empty(n, tok, hd, dtype=dtype).pin_memory())
        e.synchronize()
        assert torch.equal(g, kv.index_select(0, idx).cpu())
        dst = torch.zeros(blocks, tok, hd, device=DEV, dtype=dtype)
        e.scatter_(dst, idx, host, 0)
        e.synchronize()
        assert torch.equal(dst.index_select(0, idx), host.to(DEV))
    assert seen - {"native"}, seen  # copyengine's own strategies were explored


def test_device_indices_refuse_pageable_host_memory():
    idx = torch.arange(4, device=DEV)
    with pytest.raises(ce.Unsupported):
        ce.Engine(prior_only=True).copy_(torch.zeros(4, 8), ce.lazy(torch.randn(8, 8, device=DEV)).index_select(0, idx))
