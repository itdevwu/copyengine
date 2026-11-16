"""Footprints and dependencies: an op's footprint on each tensor is its access
over its units (an index the host does not know spanning its bound), not the
hull. Disjoint column slices of one pool run in one wave; overlapping ones are
ordered, and the result is the sequential one. Checked against torch."""

import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"


def _pool(layers, blocks, width, pinned):
    pool = torch.zeros(blocks, layers * width, dtype=torch.uint8)
    return pool.pin_memory() if pinned else pool.to(DEV)


@pytest.mark.parametrize("pinned", [True, False], ids=["d2h_pinned_pool", "gpu_pool"])
def test_scatters_into_column_slices_of_a_block_major_pool_are_one_wave(pinned):
    # The layout of the dev2 report: [blocks, sum(layer bytes)], one column slice per layer.
    L, B, W, n = 12, 32, 256, 6
    pool = _pool(L, B, W, pinned)
    caches = [torch.randint(0, 255, (n, W), dtype=torch.uint8, device=DEV) for _ in range(L)]
    ids = torch.randperm(B, device=DEV)[:n]
    e = ce.Engine(prior_only=True)
    with e.scope():
        for l, c in enumerate(caches):
            e.scatter_(pool[:, l * W:(l + 1) * W], ids, c, 0)
    assert e.stats()["last"]["waves"] == 1, e.stats()["last"]
    torch.cuda.synchronize()
    want = _pool(L, B, W, False).cpu()
    for l, c in enumerate(caches):
        want[ids.cpu(), l * W:(l + 1) * W] = c.cpu()
    assert torch.equal(pool.cpu(), want)


def test_overlapping_column_slices_are_ordered():
    B, n = 32, 6
    pool = torch.zeros(B, 512, dtype=torch.uint8, device=DEV)
    a = torch.full((n, 256), 1, dtype=torch.uint8, device=DEV)
    b = torch.full((n, 256), 2, dtype=torch.uint8, device=DEV)
    ids = torch.randperm(B, device=DEV)[:n]
    e = ce.Engine(prior_only=True)
    with e.scope():
        e.scatter_(pool[:, 0:256], ids, a, 0)
        e.scatter_(pool[:, 128:384], ids, b, 0)  # overlaps columns 128:256: after the first
    assert e.stats()["last"]["waves"] == 2, e.stats()["last"]
    torch.cuda.synchronize()
    want = torch.zeros_like(pool)
    want[ids, 0:256] = 1
    want[ids, 128:384] = 2
    assert torch.equal(pool, want)


def _quant_ref(x, q, s, fmt):
    e = ce.Engine(prior_only=True)
    e.quantize(x, q, s, fmt=fmt, group=32)
    torch.cuda.synchronize()


def test_tile_ops_into_column_slices_are_one_wave():
    # Not indexed at all: before, each op's footprint was the hull of its slice,
    # which spans every row of the pool, so all of them conflicted.
    L, M, G = 8, 64, 64
    xs = [torch.randn(M, G, device=DEV) for _ in range(L)]
    q = torch.zeros(M, L * G, dtype=torch.int8, device=DEV)
    s = torch.zeros(M, L * 2, dtype=torch.float32, device=DEV)
    e = ce.Engine(prior_only=True)
    with e.scope():
        for l, x in enumerate(xs):
            e.quantize(x, q[:, l * G:(l + 1) * G], s[:, 2 * l:2 * l + 2], fmt="int8", group=32)
    assert e.stats()["last"]["waves"] == 1, e.stats()["last"]
    torch.cuda.synchronize()
    for l, x in enumerate(xs):
        q1, s1 = torch.empty(M, G, dtype=torch.int8, device=DEV), torch.empty(M, 2, device=DEV)
        _quant_ref(x, q1, s1, "int8")
        assert torch.equal(q[:, l * G:(l + 1) * G], q1) and torch.equal(s[:, 2 * l:2 * l + 2], s1), l


def test_overlapping_tile_ops_are_ordered():
    M = 64
    x1, x2 = torch.randn(M, 64, device=DEV), torch.randn(M, 64, device=DEV) * 3
    q = torch.zeros(M, 96, dtype=torch.int8, device=DEV)
    s1, s2 = torch.zeros(M, 2, device=DEV), torch.zeros(M, 2, device=DEV)
    e = ce.Engine(prior_only=True)
    with e.scope():
        e.quantize(x1, q[:, 0:64], s1, fmt="int8", group=32)
        e.quantize(x2, q[:, 32:96], s2, fmt="int8", group=32)  # codes 32:64 overwritten by the second
    assert e.stats()["last"]["waves"] == 2, e.stats()["last"]
    torch.cuda.synchronize()
    want = torch.zeros_like(q)
    _quant_ref(x1, want[:, 0:64], torch.empty(M, 2, device=DEV), "int8")
    _quant_ref(x2, want[:, 32:96], torch.empty(M, 2, device=DEV), "int8")
    assert torch.equal(q, want)


def test_scales_written_by_one_op_and_read_by_another_are_ordered():
    # A quantize writes scales that a dequantize in the same scope reads: the
    # scale tensor is a further tensor of both ops, and its footprint orders them.
    M = 64
    x = torch.randn(M, 64, device=DEV)
    q = torch.zeros(M, 64, dtype=torch.int8, device=DEV)
    s = torch.zeros(M, 2, device=DEV)
    y = torch.zeros(M, 64, device=DEV)
    e = ce.Engine(prior_only=True)
    with e.scope():
        e.quantize(x, q, s, fmt="int8", group=32)
        e.dequantize(q, s, y, fmt="int8", group=32)
    assert e.stats()["last"]["waves"] == 2, e.stats()["last"]
    torch.cuda.synchronize()
    want = torch.empty_like(y)
    e2 = ce.Engine(prior_only=True)
    e2.dequantize(q, s, want, fmt="int8", group=32)
    torch.cuda.synchronize()
    assert torch.equal(y, want)
