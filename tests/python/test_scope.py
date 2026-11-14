"""Scope semantics, run under both interception implementations.

The C++ shim and the Python mode must behave identically: the same tests,
and a differential test comparing both with plain torch.
"""

import numpy as np
import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
CUDA = torch.cuda.is_available()
DEVS = ["cpu"] + (["cuda"] if CUDA else [])


def _impls():
    out = [pytest.param("mode", id="mode")]
    out.append(pytest.param("shim", id="shim", marks=pytest.mark.skipif(ce._shim.lib is None, reason=ce._shim.error)))
    return out


@pytest.fixture(params=_impls())
def eng(request):
    return ce.Engine(prior_only=True, _intercept=request.param)


@pytest.mark.parametrize("dev", DEVS)
def test_copies_are_queued_and_merged(eng, dev):
    L, B, N = 4, 8, 32
    slot = torch.randn(L, B, N)
    cache = torch.zeros(L, 3 * B, N, device=dev)
    ids = torch.randperm(3 * B)[:B].tolist()
    with eng.scope():
        for l in range(L):
            for i, b in enumerate(ids):
                cache[l, b].copy_(slot[l, i])  # select views are not intercepted: nothing flushes early
        assert eng._e.queued == L * B
    assert eng.stats()["last"]["ops"] == L * B
    torch.testing.assert_close(cache[:, ids].cpu(), slot, rtol=0, atol=0)


@pytest.mark.parametrize("dev", DEVS)
def test_ops_inside_a_scope_see_queued_copies(eng, dev):
    src = torch.arange(64.0, device=dev)
    dst = torch.zeros(64, device=dev)
    with eng.scope():
        dst.copy_(src)
        assert eng._e.queued == 1
        total = dst.sum()  # reads dst: the queued copy is issued first
        assert eng._e.queued == 0
        assert total.item() == 64 * 63 / 2


@pytest.mark.parametrize("dev", DEVS)
def test_writes_after_a_queued_copy_do_not_leak_into_it(eng, dev):
    src = torch.arange(64.0, device=dev)
    dst = torch.zeros(64, device=dev)
    with eng.scope():
        dst.copy_(src)
        src.add_(1000)  # writes what the queued copy reads
    torch.testing.assert_close(dst.cpu(), torch.arange(64.0), rtol=0, atol=0)


@pytest.mark.parametrize("dev", DEVS)
def test_unrelated_ops_do_not_flush(eng, dev):
    src = torch.arange(64.0, device=dev)
    dst = torch.zeros(64, device=dev)
    other = torch.ones(8, device=dev)
    with eng.scope():
        dst.copy_(src)
        other.mul_(3)
        _ = other + 1
        assert eng._e.queued == 1
    torch.testing.assert_close(dst, src, rtol=0, atol=0)


@pytest.mark.parametrize("dev", DEVS)
def test_torch_semantics_where_ce_differs(eng, dev):
    x = torch.tensor([3.7, -3.7, 1e10, -1e10, float("nan")], device=dev)
    want = torch.empty(5, dtype=torch.int32, device=dev).copy_(x)  # float -> int: torch's own result
    got = torch.empty(5, dtype=torch.int32, device=dev)
    with eng.scope():
        got.copy_(x)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert eng.stats()["ops"] == 0


@pytest.mark.parametrize("dev", DEVS)
def test_torch_errors_are_torch_errors(eng, dev):
    y = torch.arange(16.0, device=dev)
    with pytest.raises(RuntimeError, match="single memory location") as plain:
        y[1:].copy_(y[:-1])
    with pytest.raises(RuntimeError, match="single memory location") as scoped:
        with eng.scope():
            y[1:].copy_(y[:-1])  # ce refuses the overlap, so torch decides, as it would
    assert type(scoped.value) is type(plain.value)


def test_grad_tensors_go_native(eng):
    w = torch.zeros(4, requires_grad=True)
    x = torch.arange(4.0, requires_grad=True)
    with eng.scope():
        y = w.clone()
        y.copy_(x)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones(4))
    assert eng.stats()["ops"] == 0


def test_version_counter_is_bumped(eng):
    a, b = torch.zeros(8), torch.ones(8)
    v = a._version
    with eng.scope():
        a.copy_(b)
    assert a._version == v + 1


def test_nothing_is_intercepted_outside_a_scope(eng):
    a, b = torch.zeros(8), torch.ones(8)
    a.copy_(b)
    assert eng.stats()["ops"] == 0


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("non_blocking", [False, True])
def test_h2d_keeps_the_callers_blocking_semantics(eng, non_blocking):
    src = torch.randn(1 << 16).pin_memory()
    dst = torch.zeros(1 << 16, device="cuda")
    with eng.scope():
        dst.copy_(src, non_blocking=non_blocking)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst.cpu(), src, rtol=0, atol=0)


@pytest.mark.parametrize("dev", DEVS)
def test_differential_random_programs(dev):
    """Random copies and ops: the shim, the mode and plain torch agree bit for bit."""
    impls = ["mode"] + (["shim"] if ce._shim.lib is not None else [])
    rng = np.random.default_rng(1)
    for trial in range(40):
        prog = []
        for _ in range(int(rng.integers(1, 10))):
            kind = int(rng.integers(0, 4))
            a, b = (int(v) for v in rng.integers(0, 3, 2))
            n = int(rng.integers(1, 12))
            so, do = (int(v) for v in rng.integers(0, 48 - n, 2))
            prog.append((kind, a, b, n, so, do))

        def run(intercept):
            bufs = [torch.arange(48.0, device=dev) * (k + 1) for k in range(3)]
            e = ce.Engine(prior_only=True, _intercept=intercept) if intercept else None
            ctx = e.scope() if e else _null()
            with ctx:
                for kind, a, b, n, so, do in prog:
                    if kind == 0:
                        if a == b and abs(so - do) < n:
                            continue
                        bufs[b][do:do + n].copy_(bufs[a][so:so + n])
                    elif kind == 1:
                        bufs[a][so:so + n].mul_(2)
                    elif kind == 2:
                        bufs[b][do:do + n].copy_(bufs[a][so:so + n].flip(0))
                    else:
                        bufs[b][do] = bufs[a][so:so + n].sum()
            return [x.cpu() for x in bufs]

        want = run(None)
        for impl in impls:
            for g, w in zip(run(impl), want):
                torch.testing.assert_close(g, w, rtol=0, atol=0, msg=f"{impl} trial {trial}: {prog}")


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def test_shim_compiles_from_the_installed_sources(tmp_path):
    import ctypes
    out = tmp_path / "libcopyengine_shim.so"
    ce._shim.build(out)
    lib = ctypes.CDLL(str(out))
    for sym in ("copyengine_shim_init", "copyengine_shim_enter", "copyengine_shim_exit", "copyengine_shim_disable"):
        assert hasattr(lib, sym)


def test_python_mode_site_switch_keeps_torch_unless_the_mode_clearly_wins():
    from copyengine._api import _Site
    slow = _Site()
    while slow.decided is None:
        kind, timed = slow.choose()
        assert timed
        slow.report(kind == "mode", 2.0 if kind == "mode" else 1.0)
    assert slow.decided is False and slow.choose() == (None, False)
    fast = _Site()
    while fast.decided is None:
        kind, _ = fast.choose()
        fast.report(kind == "mode", 0.5 if kind == "mode" else 1.0)
    assert fast.decided is True and fast.choose() == ("mode", False)


@pytest.mark.parametrize("dev", DEVS)
def test_declined_copies_are_counted_with_a_reason(eng, dev):
    f = torch.randn(16, device=dev)
    with eng.scope():
        torch.empty(16, dtype=torch.int32, device=dev).copy_(f)  # float -> int: torch's result differs
        torch.empty(16, dtype=torch.int32, device=dev).copy_(f)
        torch.empty(16, device=dev).copy_(f)                     # taken
    assert eng.stats()["declined"] == {"dtype pair where torch's result differs": 2}
    y = torch.arange(16.0, device=dev)
    with pytest.raises(RuntimeError):
        with eng.scope():
            y[1:].copy_(y[:-1])
    assert eng.stats()["declined"]["overlapping source and destination"] == 1
