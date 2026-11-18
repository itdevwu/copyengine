"""Step plans: traced once per argument layout, replayed with new CUDA index
tensors, checked against torch."""

import numpy as np
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


def test_a_swap_through_a_temporary_replays_right():
    def step(a, b):
        tmp = torch.empty_like(a)
        tmp.copy_(a)
        a.copy_(b)
        b.copy_(tmp)

    p = ce.Engine().plan(step)
    a = torch.full((64, 64), 1.0, device=DEV)
    b = torch.full((64, 64), 2.0, device=DEV)
    for k in range(200):
        a0, b0 = a.clone(), b.clone()
        p(a, b)
        torch.cuda.synchronize()
        assert torch.equal(a, b0) and torch.equal(b, a0), k


def test_column_slices_of_one_pool_stay_one_pass():
    # 8 layers as column slices of one pinned pool (bench make_views): source layer A and
    # destination layer B share the storage, not the bytes, whatever the block ids.
    import numpy as np
    L, B, W = 8, 32, 64
    pool = torch.randn(B, L * W).pin_memory()
    dev = torch.zeros(B, W, device=DEV)
    e = ce.Engine()
    for _ in range(3):
        ids = np.random.default_rng(0).permutation(B)[:6]
        for a, b in [(0, 1), (3, 5)]:
            src = ce.lazy(pool[:, a * W:(a + 1) * W]).index_select(0, ids)
            dst = ce.lazy(pool[:, b * W:(b + 1) * W]).index_select(0, ids)
            want = pool.clone()
            want[ids, b * W:(b + 1) * W] = want[ids, a * W:(a + 1) * W]
            e.copy_(dst, src)
            torch.cuda.synchronize()
            assert torch.equal(pool, want)
            assert e.stats()["last"]["ops"] == 1  # one pass: no temporary
    del dev


def test_column_slices_with_run_time_indices_stay_one_pass():
    # GPU indices: only the conservative footprint is known. It keeps the strides (the
    # indexed dimension spans its bound, the columns stay exact), so layer A's columns and
    # layer B's columns of one pool do not overlap: one pass, no temporary.
    from copyengine import _core
    from copyengine._tensor import view
    L, B, W = 8, 32, 64
    pool = torch.randn(B, L * W, device=DEV)
    ids = torch.randperm(B, device=DEV)[:6]
    src = ce.lazy(pool[:, 0:W]).index_select(0, ids)
    dst = ce.lazy(pool[:, 3 * W:4 * W]).index_select(0, ids)
    d, s = view(dst, writable=True), view(src)
    assert _core.Engine.stmt_overlap(s.endpoint, d.endpoint, _core.Stmt(d.access, s.access)) == 0
    same = ce.lazy(pool[:, 0:W]).index_select(0, ids.flip(0))  # the same columns: they may overlap
    assert _core.Engine.stmt_overlap(view(same).endpoint, d.endpoint, _core.Stmt(view(same, writable=True).access,
                                                                                  s.access)) == 2
    want = pool.clone()
    want[ids, 3 * W:4 * W] = want[ids, 0:W]
    e = ce.Engine()
    e.copy_(dst, src)
    torch.cuda.synchronize()
    assert torch.equal(pool, want) and e.stats()["last"]["ops"] == 1


@pytest.mark.parametrize("where", ["cuda", "pinned"])
def test_overlapping_block_copies_in_one_pool_match_torch(where):
    import numpy as np
    pool = torch.randn(32, 4, 16)
    pool = pool.to(DEV) if where == "cuda" else pool.pin_memory()
    rng = np.random.default_rng(3)
    e = ce.Engine()
    for k in range(30):
        d, s = rng.permutation(32)[:8], rng.permutation(32)[:8]  # the id sets intersect most of the time
        want = pool.clone()
        want[torch.as_tensor(d)] = want[torch.as_tensor(s)]
        di = torch.as_tensor(d, device=DEV) if where == "cuda" and k % 2 else d  # GPU or host indices
        si = torch.as_tensor(s, device=DEV) if where == "cuda" and k % 2 else s
        e.copy_(ce.lazy(pool).index_select(0, di), ce.lazy(pool).index_select(0, si))
        torch.cuda.synchronize()
        assert torch.equal(pool.cpu(), want.cpu()), k


def test_step_plan_host_indices_one_or_two_passes_per_call():
    import numpy as np
    pool = torch.randn(32, 4, 16, device=DEV)
    e = ce.Engine()

    def step(pool, d, s):
        e.copy_(ce.lazy(pool).index_select(0, d), ce.lazy(pool).index_select(0, s))

    p = e.plan(step)
    rng = np.random.default_rng(5)
    for k in range(200):
        if k % 3:  # disjoint id sets: one pass
            perm = rng.permutation(32)
            d, s = perm[:8], perm[8:16]
        else:      # intersecting: two passes
            d, s = rng.permutation(32)[:8], rng.permutation(32)[:8]
        want = pool.clone()
        want[torch.as_tensor(d, device=DEV)] = want[torch.as_tensor(s, device=DEV)]
        p(pool, d, s)
        torch.cuda.synchronize()
        assert torch.equal(pool, want), k
    tr = next(iter(p._layouts.values()))["traces"][0]
    assert tr.plain is None and tr.checks


def test_a_copy_that_cannot_read_host_index_parameters_runs_once_with_values():
    # A copy the engine cannot run with a host index parameter (host to host: no GPU reads
    # the table) takes the values instead; the step is not run again (a swap run twice is
    # undone: fuzzer seed 17 with two GPUs).
    e = ce.Engine()
    gpu = torch.arange(8.0, device=DEV)
    a, b = torch.arange(16.0), torch.zeros(16)

    def step(gpu, a, b, ids):
        tmp = torch.empty(4, device=DEV)
        e.copy_(tmp, gpu[:4])
        e.copy_(gpu[:4], gpu[4:])
        e.copy_(gpu[4:], tmp)
        e.copy_(ce.lazy(b)[ids], ce.lazy(a)[ids])

    p = e.plan(step)
    for it in range(8):
        ids = np.random.default_rng(it).permutation(16)[:5]
        want_gpu = torch.cat([gpu[4:], gpu[:4]])
        want_b = b.clone()
        want_b[torch.from_numpy(ids)] = a[torch.from_numpy(ids)]
        p(gpu, a, b, ids)
        torch.cuda.synchronize()
        assert torch.equal(gpu, want_gpu) and torch.equal(b, want_b), it


@pytest.mark.parametrize("op", ["gather", "scatter_"])
def test_host_index_parameters_no_kernel_can_read_take_values_once(op):
    # Pageable host memory and a conversion: no strategy reads a host table here (the
    # statement kernel cannot reach pageable memory; DMA does not convert). Refused when
    # the copy is made, so the traced copy takes the values and the step runs once
    # (codex review, dev10: refused at the flush, the trace failed).
    e = ce.Engine()
    fn = ce.Elementwise(scale=2.0)
    host, gpu = torch.arange(64 * 8, dtype=torch.float32).view(64, 8), torch.zeros(64, 8, device=DEV)
    calls = []

    def step(ids):
        calls.append(1)
        if op == "gather":
            e.gather(host, ids, 0, out=gpu[:8], fn=fn)
        else:
            e.scatter_(host, ids, gpu[:8], 0, fn=fn)

    p = e.plan(step)
    for it in range(6):
        ids = np.random.default_rng(it).permutation(64)[:8]
        gpu.normal_()
        before = host.clone()
        n = len(calls)
        p(ids)
        torch.cuda.synchronize()
        assert len(calls) <= n + 1, it
        t = torch.from_numpy(ids)
        if op == "gather":
            assert torch.equal(gpu[:8].cpu(), host[t] * 2), it
        else:
            before[t] = gpu[:8].cpu() * 2
            assert torch.equal(host, before), it


def test_a_replay_whose_index_argument_lies_where_the_step_writes_runs_plainly():
    # Traced with an index tensor apart from what the step writes; replayed with the
    # index being the tensor a later copy overwrites. The frozen waves never ordered
    # that (codex review, dev10): the step runs plainly, the gather reads the old ids.
    e = ce.Engine()
    src = torch.arange(64.0, device=DEV).view(64, 1).expand(64, 8).contiguous()
    out = torch.zeros(4, 8, device=DEV)
    scratch, other = torch.zeros(1024, device=DEV), torch.ones(1024, device=DEV)
    b, new = torch.tensor([10, 11, 12, 13], device=DEV), torch.tensor([50, 51, 52, 53], device=DEV)

    calls = []

    def step(ids):
        calls.append(1)
        e.copy_(scratch, other)
        e.gather(src, ids, 0, out=out)
        e.copy_(b, new)

    p = e.plan(step)
    lay = None
    for _ in range(300):  # until the engine settled and the step froze
        b.copy_(torch.tensor([10, 11, 12, 13], device=DEV))
        p(torch.tensor([1, 2, 3, 4], device=DEV))
        lay = next(iter(p._layouts.values()))
        if lay["traces"]:
            break
    tr = lay["traces"][0]
    assert tr.frozen is not None and tr.plain is None, (tr.plain, e.stats()["locks"])
    tr.site.decided = True  # replays from now on (the A/B's outcome is not this test's subject)
    for _ in range(3):
        b.copy_(torch.tensor([10, 11, 12, 13], device=DEV))
        out.zero_()
        n = len(calls)
        p(b)
        torch.cuda.synchronize()
        assert len(calls) == n + 1  # ran plainly, once
        assert torch.equal(out[:, 0], torch.tensor([10.0, 11, 12, 13], device=DEV)), e.stats()["last"]
        assert torch.equal(b, new)
    n = len(calls)
    p(torch.tensor([5, 6, 7, 8], device=DEV))  # an index apart from the writes still replays
    torch.cuda.synchronize()
    assert len(calls) == n  # replayed: the step's Python did not run
    assert torch.equal(out[:, 0], torch.tensor([5.0, 6, 7, 8], device=DEV))
