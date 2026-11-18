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


def _pair(a, b):
    """The ordered pair's verdicts (a copy between them admits it): what a test may expect.
    A path the pair admitted must work: Unsupported there is a failure, not a skip."""
    e = ce.Engine()
    e.copy_(torch.empty(256, device=f"cuda:{b}"), torch.ones(256, device=f"cuda:{a}"))
    torch.cuda.synchronize(b)
    return next(c for c in e.stats()["cuda"] if c["index"] == a)["peers"][str(b)]


def _streams(a, b, default):
    if default:
        return torch.cuda.default_stream(a), torch.cuda.default_stream(b)
    return torch.cuda.Stream(a), torch.cuda.Stream(b)


@pytest.mark.parametrize("default", [True, False])
@pytest.mark.parametrize("a,b", PAIRS)
@pytest.mark.parametrize("force", [None] + FORCED)
def test_waits_for_both_gpus_and_orders_their_later_work(a, b, default, force):
    pair = _pair(a, b)
    if force in ("batch", "kernel") and not pair[force]:
        pytest.skip(f"the pair did not admit {force}: {pair.get('error', '')}")
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
        e.copy_(dst, src[:, ::2])         # a strided source: several rows
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
    if not _pair(a, b)["kernel"]:  # only the kernel converts between two GPUs: refused, said so
        with pytest.raises(ce.Unsupported):
            e.copy_(torch.zeros(64, 150, dtype=dt, device=f"cuda:{b}"), src[:, ::2])
        return
    for _ in range(10):
        dst = torch.zeros(64, 150, dtype=dt, device=f"cuda:{b}")
        e.copy_(dst, src[:, ::2])
        torch.cuda.synchronize(b)
        want = torch.zeros_like(dst)
        want.copy_(src[:, ::2])
        assert torch.equal(dst.view(torch.int16).cpu(), want.view(torch.int16).cpu())


@pytest.mark.parametrize("a,b", PAIRS)
def test_gpu_index_gather_and_scatter_between_gpus(a, b):
    # Where the pair admitted kernels the statement kernel runs them; elsewhere torch's own
    # copy may, or they are refused: never a wrong result, and never refused where admitted.
    admitted = _pair(a, b)["kernel"]
    e = ce.Engine()
    pool = torch.randn(256, 64, device=f"cuda:{a}")
    for _ in range(10):
        ids = torch.randperm(256, device=f"cuda:{a}")[:32]
        out = torch.zeros(32, 64, device=f"cuda:{b}")
        try:
            e.gather(pool, ids, 0, out=out)         # indices on the source GPU (KV migration)
        except ce.Unsupported:
            if admitted:
                raise
            return
        torch.cuda.synchronize(b)
        assert torch.equal(out.cpu(), pool[ids].cpu())
        far = torch.zeros(256, 64, device=f"cuda:{b}")
        ids_b = ids.to(f"cuda:{b}")
        e.scatter_(far, ids_b, pool[:32], 0)        # indices on the destination GPU
        torch.cuda.synchronize(b)
        want = torch.zeros(256, 64)
        want[ids.cpu()] = pool[:32].cpu()
        assert torch.equal(far.cpu(), want)


@pytest.mark.parametrize("force", ["native", "fused"])
@pytest.mark.parametrize("a,b", PAIRS)
def test_a_scoped_copy_keeps_the_source_stream_current_when_it_was_made(a, b, force):
    # GPU a's current stream changes between the copy and the scope's flush: the copy
    # is ordered after the work on the stream current when it was made, as torch's
    # would be (codex review, dev10).
    if force == "fused" and not _pair(a, b)["kernel"]:
        pytest.skip("the pair did not admit kernels")
    e = ce.Engine(_force=force)
    pool = torch.arange(256.0, device=f"cuda:{a}").view(256, 1).expand(256, 64).contiguous()
    ids = torch.zeros(32, dtype=torch.int64, device=f"cuda:{a}")
    new = torch.arange(100, 132, device=f"cuda:{a}")
    out = torch.zeros(32, 64, device=f"cuda:{b}")
    side = torch.cuda.Stream(a)
    torch.cuda.synchronize(a), torch.cuda.synchronize(b)
    with torch.cuda.device(b), e.scope():
        with torch.cuda.stream(side):  # GPU a's current stream while the copy is made
            with torch.cuda.device(a):
                torch.cuda._sleep(SLEEP)
                ids.zero_().add_(new)  # written late on that stream (not a copy_: the scope would queue it)
            e.gather(pool, ids, 0, out=out)
        # back to GPU a's default stream before the scope flushes
    torch.cuda.synchronize(a), torch.cuda.synchronize(b)
    assert torch.equal(out[:, 0].cpu(), torch.arange(100.0, 132.0)), e.stats()["last"]
    assert e.stats()["last"]["strategies"] == [force], e.stats()["last"]


@pytest.mark.parametrize("a,b", PAIRS)
def test_a_step_plan_gathering_by_host_indices_between_gpus_runs_its_step_once(a, b):
    # Host index tables between two GPUs are refused: the traced gather takes the values
    # instead, without running the step again (codex review, dev10).
    e = ce.Engine()
    pool = torch.randn(256, 64, device=f"cuda:{a}")
    out = torch.zeros(16, 64, device=f"cuda:{b}")
    calls = []

    def step(ids):
        calls.append(1)
        e.gather(pool, ids, 0, out=out)

    p = e.plan(step)
    for it in range(8):
        ids = torch.randperm(256)[:16]
        n = len(calls)
        p(ids)
        torch.cuda.synchronize(b)
        assert len(calls) - n <= 1, it
        assert torch.equal(out.cpu(), pool[ids.to(f"cuda:{a}")].cpu()), it


def test_pairs_report_their_facts():
    e = ce.Engine()
    e.copy_(torch.zeros(1024, device="cuda:1"), torch.ones(1024, device="cuda:0"))
    torch.cuda.synchronize(1)
    peers = next(c for c in e.stats()["cuda"] if c["index"] == 0)["peers"]
    p = peers["1"]
    assert {"access", "rank", "dma", "kernel", "batch"} <= set(p)
    assert p["dma"] is True
    assert p["access"] == torch.cuda.can_device_access_peer(0, 1)
