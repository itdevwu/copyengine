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


def test_memory_outside_the_arguments_runs_plainly():
    src = torch.randn(64, 32, device=DEV)  # captured, not passed

    def step(out, ids):
        ce.gather(src, ids, 0, out=out)

    p = ce.Engine().plan(step)
    out = torch.empty(8, 32, device=DEV)
    for _ in range(500):  # until the engine settled and the step was traced
        ids = torch.randperm(64, device=DEV)[:8]
        p(out, ids)
        torch.cuda.synchronize()
        assert torch.equal(out, src[ids])
        if next(iter(p._layouts.values()))["traces"]:
            break
    tr = next(iter(p._layouts.values()))["traces"][0]
    assert tr.plain and "not an argument" in tr.plain


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
