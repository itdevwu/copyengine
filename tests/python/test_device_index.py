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
