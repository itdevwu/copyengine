"""Explicit copies into host memory return with the data there, whichever path
ran (torch's baseline included): the host may read it right away. The GPU is
kept busy first, so a copy that is still in flight shows up as wrong data."""

import numpy as np
import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"


def engines():
    # The torch baseline only, and ce's own strategies while exploring.
    return [ce.Engine(disable=True), ce.Engine(tune_trials=1, tune_topk=4)]


def busy():
    torch.cuda._sleep(200_000_000)


@pytest.mark.parametrize("pinned", [True, False])
def test_lazy_index_destination_on_the_host(pinned):
    src = torch.randn(16, 256, device=DEV)
    for e in engines():
        for _ in range(12):
            base = torch.zeros(64, 256)
            base = base.pin_memory() if pinned else base
            ids = np.random.permutation(64)[:16]
            busy()
            e.copy_(ce.lazy(base).index_select(0, ids), src)
            got = base[torch.from_numpy(ids)].clone()  # read at once, no synchronize
            torch.cuda.synchronize()
            assert torch.equal(got, src.cpu())


@pytest.mark.parametrize("pinned", [True, False])
def test_views_gathers_segments_on_the_host(pinned):
    src = torch.randn(64, 256, device=DEV)
    for e in engines():
        for _ in range(12):
            out = torch.zeros(16, 256)
            out = out.pin_memory() if pinned else out
            ids = np.random.permutation(64)[:16]
            busy()
            e.gather(src, ids, 0, out=out)
            got = out.clone()
            torch.cuda.synchronize()
            assert torch.equal(got, src.cpu()[torch.from_numpy(ids)])

            dst = torch.zeros(64 * 256)
            dst = dst.pin_memory() if pinned else dst
            n = np.full(80, 64)
            so, do = np.random.permutation(256)[:80] * 64, np.arange(80) * 64
            busy()
            e.segments(src, dst, so, do, n)
            got = dst[: 80 * 64].clone()
            torch.cuda.synchronize()
            want = src.reshape(-1).cpu().view(-1, 64)[torch.from_numpy(so // 64)].reshape(-1)
            assert torch.equal(got, want)

            view = torch.zeros(256, 64)
            view = view.pin_memory() if pinned else view
            busy()
            e.copy_(view, src.t())
            got = view.clone()
            torch.cuda.synchronize()
            assert torch.equal(got, src.t().cpu())
