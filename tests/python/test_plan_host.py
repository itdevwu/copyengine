"""Step plans with host index arguments: the indices are parameters (numpy
arrays, lists of ints, CPU integer tensors), read when the copies run; both
strategies for them (DMA expanded per run, statement kernel) are candidates."""

import numpy as np
import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"


def run_plan(p, make_args, check, calls=200):
    """Calls the plan with new indices each time until it has replayed for a while."""
    seen = set()
    replays = 0
    for k in range(calls):
        args = make_args(k)
        p(*args)
        torch.cuda.synchronize()
        check(*args)
        seen.update(p._engine.stats()["last"]["strategies"])
        lay = next(iter(p._layouts.values()))
        if lay["traces"] and lay["traces"][-1].site is not None and lay["traces"][-1].site.decided is not None:
            replays += 1
            if replays > 20:
                break
    return seen


@pytest.mark.parametrize("kind", ["numpy", "list", "tensor"])
def test_h2d_gather_from_pinned_layers(kind):
    layers = [torch.randn(64, 16, 32).pin_memory() for _ in range(3)]
    pool = torch.empty(3, 8, 16, 32, device=DEV)

    def step(layers, pool, ids):
        for l, cache in enumerate(layers):
            ce.gather(cache, ids, 0, out=pool[l])

    def make(k):
        ids = np.random.default_rng(k).permutation(64)[:8]
        ids = ids.tolist() if kind == "list" else torch.from_numpy(ids) if kind == "tensor" else ids
        return layers, pool, ids

    def check(layers, pool, ids):
        i = torch.as_tensor(np.asarray(ids))
        assert torch.equal(pool.cpu(), torch.stack([c[i] for c in layers]))

    p = ce.Engine(tune_trials=1, tune_topk=4).plan(step)
    seen = run_plan(p, make, check)
    lay = next(iter(p._layouts.values()))
    assert len(p._layouts) == 1 and len(lay["traces"]) == 1, "new index values traced again"
    tr = lay["traces"][0]
    assert tr.plain is None and tr.params and not tr.vguard
    assert {"lowered", "fused"} & seen


def test_d2h_scatter_into_pinned_cache():
    cache = torch.zeros(64, 256).pin_memory()
    src = torch.randn(8, 256, device=DEV)

    def step(cache, src, ids):
        ce.scatter_(cache, ids, src, 0)

    def make(k):
        return cache, src, np.random.default_rng(k).permutation(64)[:8]

    def check(cache, src, ids):
        assert torch.equal(cache[torch.from_numpy(ids)], src.cpu())

    run_plan(ce.Engine(tune_trials=1, tune_topk=4).plan(step), make, check)


def test_conversion_takes_the_kernel():
    cache = torch.randn(64, 128).half().pin_memory()
    out = torch.empty(8, 128, device=DEV)

    def step(cache, out, ids):
        ce.gather(cache, ids, 0, out=out)  # half -> float: DMA cannot convert

    def make(k):
        return cache, out, np.random.default_rng(k).permutation(64)[:8]

    def check(cache, out, ids):
        assert torch.equal(out.cpu(), cache[torch.from_numpy(ids)].float())

    seen = run_plan(ce.Engine(tune_trials=1, tune_topk=4).plan(step), make, check)
    assert "lowered" not in seen


def test_indices_derived_inside_the_step_are_guarded_by_value():
    cache = torch.randn(128, 64).pin_memory()
    out = torch.empty(8, 64, device=DEV)

    def step(cache, out, ids):
        ce.gather(cache, np.asarray(ids) * 2, 0, out=out)  # computed: not a parameter

    def make(k):
        return cache, out, np.random.default_rng(k % 3).permutation(64)[:8]

    def check(cache, out, ids):
        assert torch.equal(out.cpu(), cache[torch.from_numpy(ids * 2)])

    p = ce.Engine().plan(step)
    for k in range(60):
        args = make(k)
        p(*args)
        torch.cuda.synchronize()
        check(*args)
    for lay in p._layouts.values():
        for tr in lay["traces"]:
            assert not tr.params and tr.vguard


def test_out_of_range_host_index_raises():
    cache = torch.randn(16, 64).pin_memory()
    out = torch.empty(4, 64, device=DEV)
    p = ce.Engine().plan(lambda cache, out, ids: ce.gather(cache, ids, 0, out=out))
    for k in range(60):
        p(cache, out, np.arange(4) + k % 4)
    torch.cuda.synchronize()
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        p(cache, out, np.array([0, 1, 99, 2]))
        torch.cuda.synchronize()
        ce.synchronize() if hasattr(ce, "synchronize") else p._engine.synchronize()
