"""Step plans: traced once per argument layout, replayed with new CUDA index
tensors, checked against torch."""

import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"


def make(layers=4, blocks=64, n=8):
    caches = [torch.randn(blocks, 16, 64, device=DEV) for _ in range(layers)]
    pool = torch.empty(layers, n, 16, 64, device=DEV)
    return caches, pool


def test_replays_with_new_index_tensors():
    caches, pool = make()
    p = ce.Engine().plan(lambda caches, pool, ids: [ce.gather(c, ids, 0, out=pool[i]) for i, c in enumerate(caches)])
    for step in range(60):
        ids = torch.randperm(64, device=DEV)[:8]  # a new tensor every step
        pool.zero_()
        p(caches, pool, ids)
        torch.cuda.synchronize()
        want = torch.stack([c[ids] for c in caches])
        assert torch.equal(pool, want), step
    tr = next(iter(p._layouts.values()))["traces"][0]
    assert tr.plain is None and tr.frozen is not None


def test_new_shapes_and_buffers_trace_again():
    caches, pool = make()
    p = ce.Engine().plan(lambda caches, out, ids: [ce.gather(c, ids, 0, out=out[i]) for i, c in enumerate(caches)])
    for n in (8, 4, 8):
        out = torch.empty(4, n, 16, 64, device=DEV)  # a new buffer each time: identity guard
        for _ in range(12):
            ids = torch.randperm(64, device=DEV)[:n]
            p(caches, out, ids)
            torch.cuda.synchronize()
            assert torch.equal(out, torch.stack([c[ids] for c in caches]))


def _until_traced(p, call, check, limit=500):
    for _ in range(limit):  # until the engine settled and the step was traced
        call()
        torch.cuda.synchronize()
        check()
        lay = next(iter(p._layouts.values()))
        if lay["traces"]:
            return lay["traces"][0]
    raise AssertionError("never traced")


def test_a_tensor_the_step_names_replays_while_it_is_the_same_view():
    src = torch.randn(64, 32, device=DEV)  # captured, not passed

    def step(out, ids):
        ce.gather(src, ids, 0, out=out)

    p = ce.Engine().plan(step)
    out = torch.empty(8, 32, device=DEV)
    box = {}

    def call():
        box["ids"] = torch.randperm(64, device=DEV)[:8]
        p(out, box["ids"])

    tr = _until_traced(p, call, lambda: torch.equal(out, src[box["ids"]]) or pytest.fail("wrong copy"))
    assert tr.plain is None and tr.consts, tr.plain
    for _ in range(5):  # replays
        call()
        torch.cuda.synchronize()
        assert torch.equal(out, src[box["ids"]])
    # The name now holds another tensor (and then a view of the same storage at
    # another offset): the guard traces again instead of replaying old addresses.
    for new in (torch.randn(64, 32, device=DEV), None):
        src = new if new is not None else torch.randn(80, 32, device=DEV)[16:]
        for _ in range(40):
            call()
            torch.cuda.synchronize()
            assert torch.equal(out, src[box["ids"]])


def test_a_list_the_step_names_is_guarded_per_element():
    pools = [torch.randn(64, 32, device=DEV) for _ in range(3)]

    def step(out, ids):
        for k, pool in enumerate(pools):
            ce.gather(pool, ids, 0, out=out[k])

    p = ce.Engine().plan(step)
    out = torch.empty(3, 8, 32, device=DEV)
    box = {}

    def call():
        box["ids"] = torch.randperm(64, device=DEV)[:8]
        p(out, box["ids"])

    def want():
        return torch.stack([q[box["ids"]] for q in pools])

    tr = _until_traced(p, call, lambda: torch.equal(out, want()) or pytest.fail("wrong copy"))
    assert tr.plain is None and len(tr.consts) == 3
    pools[1] = torch.randn(64, 32, device=DEV)  # replaced in place in the list
    for _ in range(40):
        call()
        torch.cuda.synchronize()
        assert torch.equal(out, want())


def test_memory_reached_through_other_objects_runs_plainly():
    holder = {"src": torch.randn(64, 32, device=DEV)}  # not a name the step uses for a tensor

    def step(out, ids):
        ce.gather(holder["src"], ids, 0, out=out)

    p = ce.Engine().plan(step)
    out = torch.empty(8, 32, device=DEV)
    box = {}

    def call():
        box["ids"] = torch.randperm(64, device=DEV)[:8]
        p(out, box["ids"])

    tr = _until_traced(p, call, lambda: torch.equal(out, holder["src"][box["ids"]]) or pytest.fail("wrong copy"))
    assert tr.plain and "neither an argument" in tr.plain


def test_intercepted_copies_in_the_step_are_planned_too():
    src = [torch.randn(32, 256).pin_memory() for _ in range(16)]
    dst = [torch.empty(32, 256, device=DEV) for _ in range(16)]

    def step(dst, src):
        for d, s in zip(dst, src):
            d.copy_(s, non_blocking=True)

    p = ce.Engine().plan(step)
    for _ in range(40):
        for s in src:
            s.normal_()
        p(dst, src)
        torch.cuda.synchronize()
        assert all(torch.equal(d.cpu(), s) for d, s in zip(dst, src))


def _traced(p):
    lay = next(iter(p._layouts.values()))
    return lay["traces"][-1] if lay["traces"] else None


def test_copies_through_a_temporary_compose_away():
    # The step stages through a tensor it allocates and drops: replays copy
    # straight from the source into the destination (no temporary).
    def step(src, dst):
        tmp = torch.empty(32, 64, device=DEV)
        tmp.copy_(src[::2])            # gather every other row into the temporary
        dst[:, 8:40].copy_(tmp.t())    # then out of it, transposed

    p = ce.Engine().plan(step)
    dst = torch.zeros(64, 48, device=DEV)
    for k in range(400):
        src = torch.randn(64, 64, device=DEV) if k == 0 else src.add_(1)
        p(src, dst)
        torch.cuda.synchronize()
        assert torch.equal(dst[:, 8:40], src[::2].t()), k
        tr = _traced(p)
        if tr is not None and k > 40:
            break
    assert tr is not None and tr.plain is None, tr.plain if tr else "not traced"
    allocs = {lo for _, _, lo, _ in tr.frozen.allocations}
    assert allocs <= {src.untyped_storage().data_ptr(), dst.untyped_storage().data_ptr()}


def test_a_temporary_kept_alive_is_not_composed():
    kept = []

    def step(src, dst):
        tmp = torch.empty(32, 64, device=DEV)
        tmp.copy_(src[::2])
        dst.copy_(tmp)
        kept.append(tmp)               # still alive after the step: not a temporary

    p = ce.Engine().plan(step)
    dst = torch.zeros(32, 64, device=DEV)
    src = torch.randn(64, 64, device=DEV)
    for k in range(300):
        p(src, dst)
        torch.cuda.synchronize()
        assert torch.equal(dst, src[::2])
        tr = _traced(p)
        if tr is not None:
            break
        kept.clear() if len(kept) > 4 else None
    # Still alive and named by the step (in `kept`): a guarded constant, not a temporary to compose away.
    assert tr is not None and tr.plain is None and tr.consts
    for _ in range(5):
        p(src, dst)
        torch.cuda.synchronize()
        assert torch.equal(dst, src[::2])


def test_a_temporary_read_before_it_is_written_is_not_composed():
    def step(src, dst):
        tmp = torch.zeros(32, 64, device=DEV)  # zeroed by torch (not a copy): its bytes are not the copies'
        tmp[:16].copy_(src[:16])
        dst.copy_(tmp)

    p = ce.Engine().plan(step)
    dst = torch.empty(32, 64, device=DEV)
    src = torch.randn(64, 64, device=DEV)
    for k in range(300):
        p(src, dst)
        torch.cuda.synchronize()
        assert torch.equal(dst[:16], src[:16]) and not dst[16:].any()
        tr = _traced(p)
        if tr is not None:
            break
    assert tr is None or tr.plain is not None


@pytest.mark.parametrize("h2d", [True, False], ids=["h2d", "d2h"])
def test_pageable_host_pools_replay_with_host_indices(h2d):
    import numpy as np

    host = [torch.randn(32, 4, 64) for _ in range(3)]  # pageable
    dev = [torch.zeros(32, 4, 64, device=DEV) for _ in range(3)]
    e = ce.Engine()

    def step(devs, hosts, di, hi):
        with e.scope():
            for d, h in zip(devs, hosts):
                if h2d:
                    e.copy_(ce.lazy(d).index_select(0, di), ce.lazy(h).index_select(0, hi))
                else:
                    e.copy_(ce.lazy(h).index_select(0, hi), ce.lazy(d).index_select(0, di))

    p = e.plan(step)
    rng = np.random.default_rng(1)
    for k in range(120):
        di, hi = rng.permutation(32)[:5], rng.permutation(32)[:5]
        p(dev, host, di, hi)
        torch.cuda.synchronize()
        for d, h in zip(dev, host):
            if h2d:
                assert torch.equal(d[di].cpu(), h[hi]), k
            else:
                assert torch.equal(h[hi], d[di].cpu()), k
    tr = next(iter(p._layouts.values()))["traces"][0]
    assert tr.plain is None and tr.params, tr.plain
