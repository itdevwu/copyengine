import numpy as np
import pytest

import copyengine as ce


def mapping(src_idx, dst_idx):
    o = np.argsort(dst_idx, kind="stable")
    return np.stack((np.asarray(dst_idx)[o], np.asarray(src_idx)[o]))


# ---- Access algebra vs numpy on arange: values are the offsets -----------------


def random_op(rng, a, x):
    """Apply one random data-movement op to Access `a` and numpy array `x`."""
    nd = x.ndim
    kind = rng.choice(["permute", "slice", "select", "flip", "index_select", "unsqueeze",
                       "expand", "reshape"])
    if nd == 0 and kind not in ("unsqueeze",):
        kind = "unsqueeze"
    if kind == "permute":
        p = rng.permutation(nd).tolist()
        return a.permute(p), x.transpose(p)
    if kind == "slice":
        d = int(rng.integers(nd))
        n = x.shape[d]
        start = None if rng.random() < 0.3 else int(rng.integers(-n - 1, n + 2))
        stop = None if rng.random() < 0.3 else int(rng.integers(-n - 1, n + 2))
        step = int(rng.choice([1, 1, 2, 3, -1, -2]))
        sl = [slice(None)] * nd
        sl[d] = slice(start, stop, step)
        return a.slice(d, start, stop, step), x[tuple(sl)]
    if kind == "select":
        d = int(rng.integers(nd))
        if x.shape[d] == 0:
            return a, x
        i = int(rng.integers(-x.shape[d], x.shape[d]))
        return a.select(d, i), np.take(x, i, axis=d)
    if kind == "flip":
        d = int(rng.integers(nd))
        return a.flip(d), np.flip(x, d)
    if kind == "index_select":
        d = int(rng.integers(nd))
        n = x.shape[d]
        if n == 0:
            return a, x
        idx = rng.integers(-n, n, size=int(rng.integers(0, 6)))
        return a.index_select(d, idx), np.take(x, idx, axis=d)
    if kind == "unsqueeze":
        d = int(rng.integers(-nd - 1, nd + 1))
        return a.unsqueeze(d), np.expand_dims(x, d)
    if kind == "expand":
        ones = [d for d in range(nd) if x.shape[d] == 1]
        if not ones:
            return a, x
        d = int(rng.choice(ones))
        n = int(rng.integers(0, 4))
        shape = list(x.shape)
        shape[d] = n
        return a.expand(d, n), np.broadcast_to(x, shape)
    # reshape: merge or split a random pair of dims; skip if not a view
    shape = list(x.shape)
    if nd >= 2 and rng.random() < 0.5:
        d = int(rng.integers(nd - 1))
        new = shape[:d] + [shape[d] * shape[d + 1]] + shape[d + 2:]
    else:
        d = int(rng.integers(nd))
        n = shape[d]
        divs = [k for k in range(1, n + 1) if n % k == 0] or [1]
        k = int(rng.choice(divs))
        new = shape[:d] + [k, n // k if k else 0] + shape[d + 1:]
    try:
        return a.reshape(new), x.reshape(new)
    except ce.NeedsMaterialize:
        return a, x


@pytest.mark.parametrize("seed", range(300))
def test_access_algebra_matches_numpy(seed):
    rng = np.random.default_rng(seed)
    shape = rng.integers(1, 5, size=int(rng.integers(1, 5))).tolist()
    x = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    a = ce.Access.contiguous(shape)
    for _ in range(int(rng.integers(1, 7))):
        a, x = random_op(rng, a, x)
        assert tuple(a.shape) == x.shape
        np.testing.assert_array_equal(a.offsets(), x.reshape(-1))
        if x.size:
            lo, hi = a.extent()
            assert lo == x.min() and hi == x.max()


def test_reshape_view_rules():
    x = np.arange(24).reshape(4, 6)
    a = ce.Access.contiguous([4, 6]).permute([1, 0])     # no view as [24]: a merge, a table of offsets
    np.testing.assert_array_equal(a.reshape([24]).offsets(), x.T.reshape(-1))
    np.testing.assert_array_equal(a.reshape([1, 24, 1]).offsets(), x.T.reshape(-1))
    b = ce.Access.contiguous([4, 6]).slice(1, 0, 6, 2)    # [4, 3] stride (6, 2): a view
    np.testing.assert_array_equal(b.reshape([2, 2, 3]).offsets(), x[:, ::2].reshape(-1))
    c = ce.Access.contiguous([4]).index_select(0, [1, 2])  # a table dimension, split into one
    np.testing.assert_array_equal(c.reshape([2, 1]).offsets(), np.array([1, 2]))
    d = ce.Access.contiguous([5, 4]).index_select(0, [4, 0, 2])  # merged across a table dimension
    np.testing.assert_array_equal(d.reshape([12]).offsets(), np.arange(20).reshape(5, 4)[[4, 0, 2]].reshape(-1))
    with pytest.raises(ce.NeedsMaterialize):
        a.reshape([3, 8])  # [6, 4] transposed into [3, 8]: splits and merges at once
    with pytest.raises(ce.NeedsMaterialize):
        c.reshape([1, 2]).reshape([2])  # fine
        ce.Access.contiguous([4]).index_select(0, [1, 2, 3, 0]).reshape([2, 2])  # splits a table


def test_errors():
    with pytest.raises(ValueError):
        ce.Access.contiguous([3]).select(0, 3)
    with pytest.raises(ValueError):
        ce.Access.contiguous([3]).index_select(0, [5])
    with pytest.raises(ValueError):
        ce.Stmt(ce.Access.contiguous([3]), ce.Access.contiguous([4]))


# ---- lowering -------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(200))
def test_lower_preserves_relation(seed):
    rng = np.random.default_rng(1000 + seed)
    shape = rng.integers(1, 5, size=int(rng.integers(1, 4))).tolist()
    x = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    src = ce.Access.contiguous(shape)
    for _ in range(int(rng.integers(0, 5))):
        src, x = random_op(rng, src, x)
    dshape = list(x.shape)
    dst = ce.Access.contiguous(dshape, offset=int(rng.integers(0, 7)))
    if rng.random() < 0.5 and len(dshape) >= 2:
        perm = rng.permutation(len(dshape)).tolist()
        inv = np.argsort(perm).tolist()
        dst = ce.Access.contiguous([dshape[p] for p in perm]).permute(inv)
    if 0 in dshape:
        assert len(ce.lower(ce.Stmt(dst, src))) == 0
        return
    runs = ce.lower(ce.Stmt(dst, src))
    np.testing.assert_array_equal(mapping(*runs.expand()), mapping(src.offsets(), dst.offsets()))


def test_lower_merges_contiguous_tail():
    # [layer, block, n] -> one run per (layer, block) row, merged where contiguous
    src = ce.Access.contiguous([2, 3, 8])
    dst = ce.Access.contiguous([2, 3, 8])
    assert len(ce.lower(ce.Stmt(dst, src))) == 1
    # gather blocks [2, 0, 1] of a [4, 8] cache into a contiguous [3, 8] buffer
    src = ce.Access.contiguous([4, 8]).index_select(0, [2, 0, 1])
    runs = ce.lower(ce.Stmt(ce.Access.contiguous([3, 8]), src))
    assert runs.src_off.tolist() == [16, 0] and runs.length.tolist() == [8, 16]


# ---- Runs / Rects -----------------------------------------------------------------


def test_from_indices_and_coalesce():
    r = ce.Runs.from_indices([0, 1, 2, 7, 8, 3], [10, 11, 12, 0, 1, 13])
    assert r.length.tolist() == [3, 2, 1]
    c = r.coalesce()
    assert c.src_off.tolist() == [7, 0] and c.dst_off.tolist() == [0, 10] and c.length.tolist() == [2, 4]


@pytest.mark.parametrize("seed", range(50))
def test_coalesce_and_2d_preserve_relation(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 30))
    w = int(rng.integers(1, 5))
    order = rng.permutation(n)
    gap = int(rng.integers(0, 3))
    src_off = order * (w + gap) + int(rng.integers(0, 5))
    dst_off = np.arange(n) * (w + int(rng.integers(0, 3)))
    r = ce.Runs.of(src_off, dst_off, np.full(n, w))
    want = mapping(*r.expand())
    c = r.coalesce()
    assert len(c) <= len(r)
    np.testing.assert_array_equal(mapping(*c.expand()), want)
    rect = c.recover_2d()
    np.testing.assert_array_equal(mapping(*rect.expand()), want)


def test_recover_2d_layers():
    # 12 layer copies with a constant pitch on both sides -> one 12-row block
    r = ce.Runs.of(np.arange(12) * 100, np.arange(12) * 64, np.full(12, 50))
    rect = r.coalesce().recover_2d()
    assert len(rect) == 1 and rect.height.tolist() == [12] and rect.src_pitch.tolist() == [100]


def test_clip_src():
    r = ce.Runs.of([0, 16], [100, 200], [16, 16])
    c = r.clip_src([0, 16], [5, 9])
    assert c.src_off.tolist() == [0, 16] and c.dst_off.tolist() == [100, 200] and c.length.tolist() == [5, 9]
    c = r.clip_src([3, 10], [10, 10])      # overlapping hints are unioned: [3, 20)
    assert c.src_off.tolist() == [3, 16] and c.length.tolist() == [13, 4]


def test_chunks():
    r = ce.Runs.of([0, 100], [0, 50], [7, 5])
    parts = r.chunks(4)
    assert [p.total for p in parts] == [4, 4, 4]
    np.testing.assert_array_equal(
        mapping(*np.concatenate([np.stack(p.expand()) for p in parts], axis=1)), mapping(*r.expand()))


def test_runs_validation():
    with pytest.raises(ValueError):
        ce.Runs.of([0], [0], [-1])
    with pytest.raises(ValueError):
        ce.Runs.of([0, 1], [0], [1, 1])
