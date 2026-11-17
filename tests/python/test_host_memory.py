"""Host memory as the driver sees it: pageable, pinned (DMA reaches it) or mapped
(kernels too, at its own address). Memory registered with cudaHostRegister may
be pinned with another device address (WSL2 does this): copyengine must then
keep kernels off it (zero-copy would fault) and still copy it right."""

import ctypes

import pytest

import copyengine as ce
from copyengine import _core

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"


@pytest.fixture
def registered():
    """A float32 host tensor over memory registered with cudaHostRegister(Portable), not allocated pinned."""
    rt = torch.cuda.cudart()
    keep = []

    def make(n):
        buf = (ctypes.c_float * n)()
        assert rt.cudaHostRegister(ctypes.addressof(buf), n * 4, 1) == rt.cudaError.success
        keep.append(buf)
        return torch.frombuffer(buf, dtype=torch.float32)

    yield make
    torch.cuda.synchronize()
    for buf in keep:
        rt.cudaHostUnregister(ctypes.addressof(buf))


def test_host_memory_kinds(registered):
    assert _core.host_memory(torch.zeros(16).data_ptr()) == "pageable"
    assert _core.host_memory(torch.zeros(16).pin_memory().data_ptr()) == "mapped"  # cudaHostAlloc: same address
    assert _core.host_memory(registered(16).data_ptr()) in ("pinned", "mapped")


def _kernels_used(e):
    return {d["strategy"] for d in e.stats()["last"]["decisions"]} & {"fused", "tile", "kernel"}


def test_registered_memory_copies_right(registered):
    h = registered(64 * 32).view(64, 32)
    h.copy_(torch.randn(64, 32))
    d = torch.zeros(64, 32, device=DEV)
    mapped = _core.host_memory(h.data_ptr()) == "mapped"
    for _ in range(40):  # exploration tries every candidate
        e = ce.Engine()
        e.copy_(d[:, ::2], h[:, 1::2])  # strided rows: fused / staged / runs candidates
        e.copy_(h[::2], d[1::2])
        torch.cuda.synchronize()
        if not mapped:
            assert not _kernels_used(e) - {"kernel"}, e.stats()["last"]["decisions"]
    ref_h = h.clone()
    assert torch.equal(d[:, ::2].cpu(), ref_h[:, 1::2])


def test_registered_memory_quantizes_or_refuses(registered):
    x = torch.randn(32, 128, device=DEV)
    q = registered(32 * 128).view(torch.int8).view(32, 4 * 128)[:, :128]  # int8 codes in registered memory
    s = registered(32).view(32, 1)
    mapped = _core.host_memory(q.data_ptr()) == "mapped"
    e = ce.Engine()
    try:
        for _ in range(20):
            e.quantize(x, q, s, fmt="int8", group=128)
            torch.cuda.synchronize()
    except ce.Unsupported as err:  # no zero-copy and not compact enough to stage: refused, not faulted
        assert not mapped, err
        return
    if not mapped:
        assert "tile" not in {d["strategy"] for d in e.stats()["last"]["decisions"]}
    q2, s2 = torch.empty(32, 128, dtype=torch.int8, device=DEV), torch.empty(32, 1, device=DEV)
    ce.Engine(prior_only=True).quantize(x, q2, s2, fmt="int8", group=128)
    torch.cuda.synchronize()
    assert torch.equal(q.clone(), q2.cpu()) and torch.equal(s.clone(), s2.cpu())


def test_device_indices_with_unmapped_memory(registered):
    h = registered(64 * 32).view(64, 32)
    h.copy_(torch.randn(64, 32))
    src = torch.randn(8, 32, device=DEV)
    idx = torch.randperm(64, device=DEV)[:8]
    ce.Engine().scatter_(h, idx, src, 0)  # host destination: the indices come to the host, DMA writes the rows
    torch.cuda.synchronize()
    assert torch.equal(h[idx.cpu()], src.cpu())
    out = torch.empty(8, 32, device=DEV)
    if _core.host_memory(h.data_ptr()) == "mapped":
        ce.copy_(out, ce.lazy(h).index_select(0, idx))  # the statement kernel reads mapped host memory
        torch.cuda.synchronize()
        assert torch.equal(out.cpu(), h[idx.cpu()])
    else:  # only the statement kernel reads GPU indices: it cannot reach this memory
        with pytest.raises(ce.Unsupported, match="not mapped"):
            ce.copy_(out, ce.lazy(h).index_select(0, idx))
