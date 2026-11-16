"""The same copies at new addresses each call (views of a pool picked per call).

The engine sees plan churn: those flushes are tuned in buckets of their own
with a strategy's setup counted (before, setup runs were no samples, so such
copies explored forever). copy_many itself is measured against
torch._foreach_copy_ per call layout and keeps whichever is faster."""

import json

import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"


def _pools(L=4, B=64, pinned=True):
    host = [torch.randn(B, 8, 32) for _ in range(L)]
    host = [h.pin_memory() for h in host] if pinned else host
    return host, [torch.zeros(B, 8, 32, device=DEV) for _ in range(L)]


def test_views_picked_per_call_settle():
    host, dev = _pools()
    B, n = 64, 3
    e = ce.Engine()
    g = torch.Generator().manual_seed(0)
    for step in range(600):
        h = torch.randperm(B, generator=g)[:n].tolist()
        d = torch.randperm(B, generator=g)[:n].tolist()
        e.copy_many([dv[j] for dv in dev for j in d], [hv[i] for hv in host for i in h])
        torch.cuda.synchronize()
        for dv, hv in zip(dev, host):
            assert torch.equal(dv[d].cpu(), hv[h]), step
        sites = e.stats()["copy_many"]
        if sites and sites[0]["copyengine"] is not None:
            break
    st = e.stats()
    assert st["churn_flushes"] > 0
    assert sites[0]["copyengine"] is not None, "copy_many never decided between copyengine and torch"
    buckets = json.loads(e.dump_profile())["buckets"]
    assert any(k.endswith("|churn") and v["locked"] for k, v in buckets.items()), list(buckets)


def test_repeated_addresses_are_no_churn():
    host, dev = _pools()
    e = ce.Engine()
    sets = [([0, 5, 9], [1, 2, 3]), ([7, 8, 4], [10, 11, 12])]  # a double buffer: two address sets
    for step in range(60):
        h, d = sets[step % 2]
        e.copy_many([dv[j] for dv in dev for j in d], [hv[i] for hv in host for i in h])
        torch.cuda.synchronize()
        for dv, hv in zip(dev, host):
            assert torch.equal(dv[d].cpu(), hv[h]), step
    assert e.stats()["churn_flushes"] == 0


def test_copy_many_left_to_torch_matches_and_freeze_refuses():
    host, dev = _pools(pinned=False)
    e = ce.Engine()
    e.copy_many(dev, host)  # a ce flush: the site is created
    torch.cuda.synchronize()
    (site,) = e._many_sites.values()
    site.decided = False  # as if torch had won the A/B
    for h in host:
        h.add_(1)
    e.copy_many(dev, host)
    for dv, hv in zip(dev, host):  # torch's copies, complete on return (pageable host side)
        assert torch.equal(dv.cpu(), hv)
    with pytest.raises(RuntimeError, match="ran as torch's own"):
        e.freeze()
    with e.scope():  # issued through copyengine: freezable again
        e.copy_many(dev, host)
    e.freeze()
