"""Explicit-API copies that torch can also express carry a torch baseline: the
A/B must beat it, and CE_DISABLE runs it. Engine(disable=True) runs exactly the
baseline, so these check that it exists and gives the reference bits."""

import numpy as np
import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def native_only():
    return ce.Engine(disable=True)


def ran(e):
    """['native'] if torch's copy ran (a lone op runs it right at interception)."""
    st = e.stats()
    if st["native_ops"]:
        return ["native"]
    return [d["strategy"] for d in st["last"]["decisions"]]


@pytest.mark.parametrize("dev", DEVS)
def test_lazy_sources(dev):
    cache = torch.randn(32, 4, 8, 16, device=dev)
    ids = [5, 0, 31, 7]
    idx = torch.tensor(ids, device=dev)
    cases = [
        (lambda: ce.lazy(cache).index_select(0, ids).permute(0, 2, 1, 3), lambda: cache[idx].permute(0, 2, 1, 3),
         torch.float16),
        (lambda: ce.lazy(cache).slice(1, 0, 4, 2).select(2, 3), lambda: cache[:, 0:4:2, 3], torch.float32),
        (lambda: ce.lazy(cache).flip(3).index_select(0, ids), lambda: cache.flip(3)[idx], torch.bfloat16),
        (lambda: ce.lazy(cache[0, 0, 0]).unsqueeze(0).expand(6, 16), lambda: cache[0, 0, 0].expand(6, 16),
         torch.float32),
        (lambda: ce.lazy(cache).reshape(32, 512).index_select(0, ids), lambda: cache.reshape(32, 512)[idx],
         torch.float64),
    ]
    for src, want, dt in cases:
        w = want().to(dt)
        out = torch.zeros(w.shape, dtype=dt, device=dev)
        e = native_only()
        e.copy_(out, src())
        assert ran(e) == ["native"]
        assert torch.equal(out, w)


@pytest.mark.parametrize("dev", DEVS)
def test_lazy_destinations(dev):
    src = torch.randn(4, 16, 8, device=dev)
    ids = [9, 2, 30, 17]
    idx = torch.tensor(ids, device=dev)
    cache = torch.zeros(32, 16, 8, dtype=torch.bfloat16, device=dev)
    e = native_only()
    e.copy_(ce.lazy(cache).index_select(0, ids), src)  # scatter by index, with a cast
    assert ran(e) == ["native"] and torch.equal(cache[idx], src.bfloat16())
    d = torch.zeros(8, 16, 4, device=dev)
    e = native_only()
    e.copy_(ce.lazy(d).permute(2, 1, 0), src)  # a view of the destination
    assert ran(e) == ["native"] and torch.equal(d.permute(2, 1, 0), src)


@pytest.mark.parametrize("dev", DEVS)
def test_gather_scatter_segments_indices(dev):
    src = torch.randn(64, 32, device=dev)
    ids = [3, 60, 1, 33, 8]
    idx = torch.tensor(ids, device=dev)
    e = native_only()
    out = e.gather(src, ids, 0, dtype=torch.float16)
    assert ran(e) == ["native"] and torch.equal(out, src[idx].half())
    dst = torch.zeros(64, 32, device=dev)
    e = native_only()
    e.scatter_(dst, ids, src[:5])
    assert ran(e) == ["native"] and torch.equal(dst[idx], src[:5])
    flat, out = src.reshape(-1), torch.zeros(64 * 32, device=dev)
    for nseg in (3, 300):  # loop of slices, then one flat index
        so = np.arange(nseg) * 6
        do = np.arange(nseg) * 6 + 1
        n = np.full(nseg, 4)
        out.zero_()
        e = native_only()
        e.segments(flat, out, so, do, n)
        want = torch.zeros_like(out)
        for a, b, k in zip(so, do, n):
            want[b:b + k] = flat[a:a + k]
        assert ran(e) == ["native"] and torch.equal(out, want), nseg
    si, di = np.array([5, 9, 100, 7]), np.array([0, 50, 2, 3])
    out.zero_()
    e = native_only()
    e.indices(flat, out, si, di)
    assert ran(e) == ["native"] and torch.equal(out[torch.tensor(di, device=dev)], flat[torch.tensor(si, device=dev)])


@pytest.mark.parametrize("dev", DEVS)
def test_no_baseline_where_torch_differs_or_cannot(dev):
    x = torch.randn(16, 8, device=dev)
    e = native_only()
    e.copy_(torch.zeros(8, 8, device=dev), ce.lazy(x).slice(0, 15, None, -2))  # negative step: no torch view
    assert ran(e) != ["native"]
    e = native_only()
    e.gather(x, [1, 2], 0, out=torch.zeros(2, 8, device=dev), fn=ce.Elementwise(scale=2.0))  # a function
    assert ran(e) != ["native"]
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    e = native_only()
    e.copy_(np.zeros((4, 4), np.float32), ce.lazy(a).index_select(0, [3, 1, 0, 2]))  # numpy: no torch replay
    assert ran(e) != ["native"]


def test_default_engine_still_correct_with_baselines():
    dev = DEVS[-1]
    cache = torch.randn(256, 8, 64, device=dev)
    ids = list(range(0, 256, 3))
    out = torch.empty(len(ids), 64, 8, dtype=torch.float16, device=dev)
    e = ce.Engine(tune_trials=1, ab_pairs=6, ab_max_pairs=10)
    for _ in range(40):
        e.copy_(out, ce.lazy(cache).index_select(0, ids).permute(0, 2, 1))
    assert torch.equal(out, cache[torch.tensor(ids, device=dev)].permute(0, 2, 1).half())
    assert "native" in e.stats()["last"]["decisions"][0]["why"] or ran(e) == ["native"]
