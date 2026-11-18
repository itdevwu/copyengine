"""Block copies within one pool by GPU indices that may overlap: torch's result
(pool[d] = pool[s] gathers first), as three kernels: mark the destination rows,
stage the source rows that are destination rows, scatter (abi.hpp: RowsOv)."""

import numpy as np
import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"


def _want(pool, d, s):
    p = pool.clone()
    p[d] = p[s]  # torch: gathers first
    return p


def _rows(e):
    return [x for x in e.stats()["last"]["decisions"] if "|rows" in x["bucket"]]


def _cases(rows, n, rng):
    perm = torch.as_tensor(rng.permutation(rows)[: 2 * n])
    yield "disjoint", perm[:n], perm[n:]
    d = torch.as_tensor(rng.permutation(rows)[:n])
    yield "chain", d, torch.roll(d, 1)               # every source row is another copy's destination
    yield "partial", d, torch.cat([d[: n // 2], perm[n: n + n - n // 2]])
    yield "repeated sources", d, torch.full((n,), int(d[0]))
    yield "identity", d, d


@pytest.mark.parametrize("shape,dtype", [((256, 16, 8), torch.bfloat16), ((97, 33), torch.float32),
                                         ((64, 3), torch.uint8)])
def test_matches_torch(shape, dtype):
    rng = np.random.default_rng(0)
    e = ce.Engine()
    n = min(24, shape[0] // 3)
    for name, d, s in _cases(shape[0], n, rng):
        for _ in range(6):  # exploration runs every candidate
            pool = (torch.randn(shape, device=DEV) * 100).to(dtype)
            want = _want(pool, d.to(DEV), s.to(DEV))
            e.copy_(ce.lazy(pool)[d.to(DEV)], ce.lazy(pool)[s.to(DEV)])
            torch.cuda.synchronize()
            assert torch.equal(pool, want), (name, shape, dtype)
        assert _rows(e), (name, e.stats()["last"])  # the overlap path, not two passes


def test_frozen_and_step_plan_replays_take_new_indices():
    rng = np.random.default_rng(1)
    pool = torch.randn(512, 64, device=DEV)
    e = ce.Engine()
    step = e.plan(lambda pool, d, s: e.copy_(ce.lazy(pool)[d], ce.lazy(pool)[s]))
    for it in range(60):
        d = torch.as_tensor(rng.permutation(512)[:40], device=DEV)
        s = torch.as_tensor(rng.integers(0, 512, 40), device=DEV)
        want = _want(pool, d, s)
        step(pool, d, s)
        torch.cuda.synchronize()
        assert torch.equal(pool, want), it


def test_two_streams_keep_their_own_marks():
    rng = np.random.default_rng(2)
    e = ce.Engine()
    pools = [torch.randn(256, 32, device=DEV) for _ in range(2)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for it in range(30):
        wants = []
        for p, st in zip(pools, streams):
            d = torch.as_tensor(rng.permutation(256)[:64], device=DEV)
            s = torch.roll(d, 1)
            wants.append(_want(p, d, s))
            with torch.cuda.stream(st):
                st.wait_stream(torch.cuda.default_stream())
                e.copy_(ce.lazy(p)[d], ce.lazy(p)[s])
        torch.cuda.synchronize()
        for p, w in zip(pools, wants):
            assert torch.equal(p, w), it


def test_an_index_out_of_range_is_reported():
    e = ce.Engine(prior_only=True)
    pool = torch.randn(16, 8, device=DEV)
    d = torch.tensor([1, 2], device=DEV)
    s = torch.tensor([3, 99], device=DEV)
    with pytest.raises((ce.CopyEngineError, ValueError)):
        e.copy_(ce.lazy(pool)[d], ce.lazy(pool)[s])
        e.synchronize()


def test_graph_replays_advance_the_epoch():
    # Captured after a run on the capturing stream: every replay marks with a new
    # epoch (kept in device memory), with whatever indices the tensors hold then.
    rng = np.random.default_rng(3)
    pool = torch.randn(128, 64, device=DEV)
    d = torch.zeros(32, dtype=torch.int64, device=DEV)
    s = torch.zeros(32, dtype=torch.int64, device=DEV)
    e = ce.Engine(_force="fused")  # the three kernels captured (else the frozen plan may replay torch's copy)
    side = torch.cuda.Stream()

    def fill():
        dd = rng.permutation(128)[:32]
        d.copy_(torch.as_tensor(dd))
        s.copy_(torch.as_tensor(np.roll(dd, 1)))

    fill()
    for _ in range(10):
        e.copy_(ce.lazy(pool)[d], ce.lazy(pool)[s])
    fr = e.freeze()
    torch.cuda.synchronize()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fr.run()  # this stream's buffers
    side.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        fr.run()
    for it in range(8):
        fill()
        torch.cuda.synchronize()
        want = _want(pool, d, s)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(pool, want), it


def test_a_graph_keeps_its_buffers_when_eager_copies_grow_them():
    # Eager copies on the capturing stream that need more marks and staging than the
    # captured graph take new buffers: the graph's stay its own (codex review, dev10).
    rng = np.random.default_rng(4)
    pool = torch.randn(128, 64, device=DEV)
    d = torch.zeros(32, dtype=torch.int64, device=DEV)
    s = torch.zeros(32, dtype=torch.int64, device=DEV)
    e = ce.Engine(_force="fused")  # the three kernels (the tuner might lock torch's copy: nothing captured)
    side = torch.cuda.Stream()

    def fill():
        dd = rng.permutation(128)[:32]
        d.copy_(torch.as_tensor(dd))
        s.copy_(torch.as_tensor(np.roll(dd, 1)))

    fill()
    for _ in range(10):
        e.copy_(ce.lazy(pool)[d], ce.lazy(pool)[s])
    fr = e.freeze()
    torch.cuda.synchronize()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fr.run()
    side.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        fr.run()
    # Larger eager batches on the same stream: more rows, two pools per flush.
    big = [torch.randn(4096, 64, device=DEV) for _ in range(2)]
    side.wait_stream(torch.cuda.current_stream())  # `big` is written on the current stream
    with torch.cuda.stream(side):
        for _ in range(4):
            wants = []
            with e.scope():
                for p in big:
                    bd = torch.as_tensor(rng.permutation(4096)[:1024], device=DEV)
                    bs = torch.roll(bd, 1)
                    wants.append(_want(p, bd, bs))
                    e.copy_(ce.lazy(p)[bd], ce.lazy(p)[bs])
            side.synchronize()
            for p, w in zip(big, wants):
                assert torch.equal(p, w), e.stats()["last"]
    # Memory freed now would be handed out again: the graph must not write into it.
    with torch.cuda.stream(side):  # torch's allocator hands freed blocks out again on their stream: fill them
        junk = [torch.zeros(512, dtype=torch.uint8, device=DEV) for _ in range(8192)]
    side.synchronize()
    for it in range(4):
        fill()
        torch.cuda.synchronize()
        want = _want(pool, d, s)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(pool, want), it
        assert not torch.cat(junk).any(), it


def test_an_sm_limit_runs_every_copy_of_a_batch():
    # The soft SM limit caps the grid's y dimension below the batch's copies (codex
    # review, dev10): each kernel strides over them. 40 pools: two batches (36 + 4).
    rng = np.random.default_rng(5)
    e = ce.Engine(max_sms=1)
    pools = [torch.randn(64, 16, device=DEV) for _ in range(40)]
    for it in range(4):
        wants = []
        with e.scope():
            for p in pools:
                d = torch.as_tensor(rng.permutation(64)[:16], device=DEV)
                s = torch.roll(d, 1)
                wants.append(_want(p, d, s))
                e.copy_(ce.lazy(p)[d], ce.lazy(p)[s])
        torch.cuda.synchronize()
        assert _rows(e) and e.stats()["last"]["calls"] == 6, e.stats()["last"]  # two batches of three kernels
        for k, (p, w) in enumerate(zip(pools, wants)):
            assert torch.equal(p, w), (it, k, e.stats()["cuda"][0].get("sm_limit"), e.stats()["last"])


def test_two_graphs_captured_on_one_stream_do_not_share_buffers():
    # The second capture finds the stream's buffers taken by the first graph: it records
    # torch's copy (or two passes) instead, and both graphs replay right, also at once
    # on two streams (codex review, dev10).
    rng = np.random.default_rng(6)
    rows, n = 1 << 16, 1 << 14  # long enough kernels that the two replays run at the same time
    pools = [torch.randn(rows, 64, device=DEV) for _ in range(2)]
    idx = [(torch.zeros(n, dtype=torch.int64, device=DEV), torch.zeros(n, dtype=torch.int64, device=DEV))
           for _ in range(2)]
    e = ce.Engine(_force="fused")
    side = torch.cuda.Stream()

    def fill():
        for d, s in idx:
            dd = rng.permutation(rows)[:n]
            d.copy_(torch.as_tensor(dd))
            s.copy_(torch.as_tensor(np.roll(dd, 1)))

    fill()
    graphs = []
    for p, (d, s) in zip(pools, idx):
        for _ in range(3):
            e.copy_(ce.lazy(p)[d], ce.lazy(p)[s])
        fr = e.freeze()
        torch.cuda.synchronize()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fr.run()
        side.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=side):
            fr.run()
        graphs.append(g)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for it in range(4):
        fill()
        torch.cuda.synchronize()
        wants = [_want(p, d, s) for p, (d, s) in zip(pools, idx)]
        for g, st in zip(graphs, streams):
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                g.replay()
        torch.cuda.synchronize()
        for p, w in zip(pools, wants):
            assert torch.equal(p, w), it


def test_an_eager_capture_takes_two_passes():
    pool = torch.randn(64, 16, device=DEV)
    d = torch.arange(8, device=DEV)
    s = torch.arange(1, 9, device=DEV)
    e = ce.Engine()
    for _ in range(3):
        e.copy_(ce.lazy(pool)[d], ce.lazy(pool)[s])
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        e.copy_(ce.lazy(pool)[d], ce.lazy(pool)[s])
    for it in range(3):
        want = _want(pool, d, s)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(pool, want), it
