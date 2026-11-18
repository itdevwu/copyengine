"""Host index values beside an index read on the GPU (one table per side for the statement
kernel): the host values go as host tables, snapshots of the values the view was built with."""

import numpy as np
import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"


@pytest.mark.parametrize("host_index", ["numpy", "list", "cpu_tensor", "negative"])
def test_gpu_indices_on_one_side_host_values_on_the_other(host_index):
    e = ce.Engine()
    gpu = torch.zeros(64, 256, device=DEV)
    host = torch.randn(32, 256).pin_memory()
    r = np.random.default_rng(0)
    for _ in range(6):
        gid = torch.from_numpy(r.permutation(64)[:8]).to(DEV)
        cid = r.permutation(32)[:8]
        idx = {"numpy": cid, "list": cid.tolist(), "cpu_tensor": torch.from_numpy(cid),
               "negative": cid - 32}[host_index]
        e.copy_(ce.lazy(gpu)[gid], ce.lazy(host)[idx])                      # H2D
        torch.cuda.synchronize()
        assert torch.equal(gpu[gid].cpu(), host[torch.from_numpy(cid)])
        back = torch.zeros(32, 256).pin_memory()
        e.copy_(ce.lazy(back)[idx], ce.lazy(gpu)[gid])                      # D2H
        torch.cuda.synchronize()
        assert torch.equal(back[torch.from_numpy(cid)], gpu[gid].cpu())


def test_duplicate_host_destination_indices_are_refused():
    e = ce.Engine()
    gpu = torch.randn(64, 32, device=DEV)
    back = torch.zeros(16, 32).pin_memory()
    with pytest.raises(ce.UnsafeCopy):
        e.copy_(ce.lazy(back)[np.array([1, 2, 2, 3])], ce.lazy(gpu)[torch.tensor([0, 1, 2, 3], device=DEV)])


def test_a_broadcast_destination_is_refused():
    # Distinct rows, but each written through a stride-0 dimension (codex review, dev10).
    e = ce.Engine()
    gpu = torch.randn(64, 8, device=DEV)
    back = torch.zeros(16, 1).pin_memory()
    with pytest.raises(ce.UnsafeCopy):
        e.copy_(ce.lazy(back).expand(16, 8)[np.array([1, 2, 3, 4])], ce.lazy(gpu)[torch.tensor([0, 1, 2, 3], device=DEV)])
    e.copy_(ce.lazy(back.expand(16, 8).contiguous().pin_memory())[np.array([1, 2, 3, 4])],  # distinct bytes: fine
            ce.lazy(gpu)[torch.tensor([0, 1, 2, 3], device=DEV)])


@pytest.mark.parametrize("kind", ["numpy", "cpu_tensor"])
def test_the_values_are_those_of_when_the_view_was_built(kind):
    e = ce.Engine()
    gpu = torch.zeros(8, 16, device=DEV)
    host = torch.arange(32 * 16, dtype=torch.float32).view(32, 16).pin_memory()
    cid = np.array([5, 6, 7, 8], dtype=np.int64)
    idx = cid if kind == "numpy" else torch.from_numpy(cid)  # a CPU tensor sharing the array's memory
    gid = torch.tensor([0, 1, 2, 3], device=DEV)
    with e.scope():
        src = ce.lazy(host)[idx]
        cid[:] = [20, 21, 22, 23]    # rewritten after the view: torch's indexing used the old values
        e.copy_(ce.lazy(gpu)[gid], src)
    torch.cuda.synchronize()
    assert torch.equal(gpu[:4].cpu(), host[[5, 6, 7, 8]])


def test_frozen_and_step_plans_with_both_kinds_of_indices():
    e = ce.Engine()
    gpu = torch.zeros(64, 128, device=DEV)
    host = torch.randn(32, 128).pin_memory()
    r = np.random.default_rng(1)
    gid = torch.from_numpy(r.permutation(64)[:8]).to(DEV)
    cid = r.permutation(32)[:8]
    e.copy_(ce.lazy(gpu)[gid], ce.lazy(host)[cid])
    fr = e.freeze()                      # replays the snapshot it was frozen with
    gpu.zero_()
    fr.run()
    torch.cuda.synchronize()
    assert torch.equal(gpu[gid].cpu(), host[torch.from_numpy(cid)])
    step = e.plan(lambda g, c: e.copy_(ce.lazy(gpu)[g], ce.lazy(host)[c]))
    for it in range(60):                 # its A/B also runs the step plainly: that must work too
        gid = torch.from_numpy(r.permutation(64)[:8]).to(DEV)
        cid = torch.from_numpy(r.permutation(32)[:8])
        step(gid, cid)
        torch.cuda.synchronize()
        assert torch.equal(gpu[gid].cpu(), host[cid]), it
