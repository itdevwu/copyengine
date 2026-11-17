"""Copies between two GPUs: ordered as torch's own (on the source GPU's current
stream, after the work queued on both GPUs' current streams and before their
later work), admitted per ordered pair, results bit for bit as torch's.

Needs two GPUs (the validation kit runs it on the machines that have them)."""

import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                                reason="needs two CUDA GPUs")

PAIRS = [(0, 1), (1, 0)]
SLEEP = 200_000_000  # cycles: long enough that an unordered copy would run first
FORCED = ["runs", "batch", "kernel", "passthrough"]


def _streams(a, b, default):
    if default:
        return torch.cuda.default_stream(a), torch.cuda.default_stream(b)
    return torch.cuda.Stream(a), torch.cuda.Stream(b)


@pytest.mark.parametrize("default", [True, False])
@pytest.mark.parametrize("a,b", PAIRS)
@pytest.mark.parametrize("force", [None] + FORCED)
def test_waits_for_both_gpus_and_orders_their_later_work(a, b, default, force):
    e = ce.Engine(_force=force)
    sa, sb = _streams(a, b, default)
    src = torch.zeros(64, 1024, device=f"cuda:{a}")
    dst = torch.zeros(64, 512, device=f"cuda:{b}")
    torch.cuda.synchronize(a), torch.cuda.synchronize(b)
    with torch.cuda.stream(sa), torch.cuda.stream(sb):  # the current stream of each GPU
        with torch.cuda.device(a):
            torch.cuda._sleep(SLEEP)
            src.fill_(7.0)            # written late on GPU a's current stream
        with torch.cuda.device(b):
            torch.cuda._sleep(SLEEP)
            dst.fill_(-1.0)           # written late on GPU b's current stream (WAW)
        try:
            e.copy_(dst, src[:, ::2])     # a strided source: several rows
        except ce.Unsupported:
            pytest.skip(f"{force} is not offered between these GPUs")
        with torch.cuda.device(b):
            dst.add_(1.0)             # later work on GPU b sees the copy
        with torch.cuda.device(a):
            src.fill_(3.0)            # later work on GPU a does not overtake the copy's read (WAR)
    torch.cuda.synchronize(a), torch.cuda.synchronize(b)
    assert torch.equal(dst.cpu(), torch.full((64, 512), 8.0))


@pytest.mark.parametrize("a,b", PAIRS)
def test_intercepted_copies_between_gpus(a, b):
    e = ce.Engine()
    src = torch.randn(32, 256, device=f"cuda:{a}")
    for _ in range(20):  # exploration runs every strategy
        dst = torch.zeros(32, 128, device=f"cuda:{b}")
        with e.scope():
            dst.copy_(src[:, 1::2])
        torch.cuda.synchronize(b)
        assert torch.equal(dst.cpu(), src[:, 1::2].cpu())


@pytest.mark.parametrize("a,b", PAIRS)
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
def test_converting_copies_between_gpus_match_torch(a, b, dt):
    e = ce.Engine()
    src = torch.randn(64, 300, device=f"cuda:{a}") * 1000
    for _ in range(10):
        dst = torch.zeros(64, 150, dtype=dt, device=f"cuda:{b}")
        try:
            e.copy_(dst, src[:, ::2])
        except ce.Unsupported as ex:
            pytest.skip(str(ex))
        torch.cuda.synchronize(b)
        want = torch.zeros_like(dst)
        want.copy_(src[:, ::2])
        assert torch.equal(dst.view(torch.int16).cpu(), want.view(torch.int16).cpu())


@pytest.mark.parametrize("a,b", PAIRS)
def test_gpu_index_gather_and_scatter_between_gpus(a, b):
    e = ce.Engine()
    pool = torch.randn(256, 64, device=f"cuda:{a}")
    for _ in range(10):
        ids = torch.randperm(256, device=f"cuda:{a}")[:32]
        out = torch.zeros(32, 64, device=f"cuda:{b}")
        try:
            e.gather(pool, ids, 0, out=out)         # indices on the source GPU (KV migration)
        except ce.Unsupported as ex:
            pytest.skip(str(ex))
        torch.cuda.synchronize(b)
        assert torch.equal(out.cpu(), pool[ids].cpu())
        far = torch.zeros(256, 64, device=f"cuda:{b}")
        ids_b = ids.to(f"cuda:{b}")
        e.scatter_(far, ids_b, 0, pool[:32])        # indices on the destination GPU
        torch.cuda.synchronize(b)
        want = torch.zeros(256, 64)
        want[ids.cpu()] = pool[:32].cpu()
        assert torch.equal(far.cpu(), want)


def test_pairs_report_their_facts():
    e = ce.Engine()
    e.copy_(torch.zeros(1024, device="cuda:1"), torch.ones(1024, device="cuda:0"))
    torch.cuda.synchronize(1)
    peers = next(c for c in e.stats()["cuda"] if c["index"] == 0)["peers"]
    p = peers["1"]
    assert {"access", "rank", "dma", "kernel", "batch"} <= set(p)
    assert p["dma"] is True
    assert p["access"] == torch.cuda.can_device_access_peer(0, 1)
