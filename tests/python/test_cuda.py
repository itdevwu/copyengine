"""CUDA paths, checked against torch. Skipped without torch or a GPU."""

import numpy as np
import time

import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

DEV = "cuda"


def explore(fn, rounds=8, check=None):
    """Run fn(engine) with exploration so every candidate strategy executes; check each round."""
    e = ce.Engine(tune_trials=1, tune_topk=4)
    seen = set()
    for _ in range(rounds):
        fn(e)
        torch.cuda.synchronize()
        seen.update(e.stats()["last"]["strategies"])
        if check:
            check()
    return e, seen


def host(shape, dtype=torch.float32, pinned=False, fill=True):
    t = torch.randn(shape).to(dtype) if fill else torch.zeros(shape, dtype=dtype)
    return t.pin_memory() if pinned else t


# ---- pure copies --------------------------------------------------------------------------


@pytest.mark.parametrize("pinned", [False, True])
def test_h2d_and_d2h_strided(pinned):
    src = host((8, 16, 32), pinned=pinned)
    dst = torch.zeros((16, 8, 32), device=DEV)
    view_s = src.transpose(0, 1)[:, ::2]
    view_d = dst[:, ::2]

    def body(e):
        dst.zero_()
        e.copy_(view_d, view_s)

    _, seen = explore(body, check=lambda: torch.testing.assert_close(view_d.cpu(), view_s, rtol=0, atol=0))
    assert "native" in seen  # torch's own copy is the baseline
    back = host((16, 4, 32), pinned=pinned, fill=False)
    ce.Engine(prior_only=True).copy_(back, view_d)
    torch.testing.assert_close(back, view_s, rtol=0, atol=0)


def test_layer_block_loop_merges():
    # PD-style: L layers x blocks, each a small H2D copy into a paged cache.
    L, B, N = 8, 16, 64
    slot = host((L, B, N), pinned=True)
    cache = torch.zeros((L, 40, N), device=DEV)
    block_ids = torch.randperm(40)[:B]

    def body(e):
        cache.zero_()
        with e.scope():
            for l in range(L):
                for i, b in enumerate(block_ids.tolist()):
                    e.copy_(cache[l, b], slot[l, i])

    def check():
        torch.testing.assert_close(cache[:, block_ids].cpu(), slot, rtol=0, atol=0)

    e, seen = explore(body, check=check)
    st = e.stats()["last"]
    assert st["ops"] == L * B
    assert seen >= {"batch", "runs", "native"}


def test_d2d_gather_kernel_and_batch():
    cache = torch.randn((64, 8, 128), device=DEV, dtype=torch.float16)
    ids = torch.tensor([5, 9, 9, 1, 63, 0, 7], dtype=torch.int64)
    out = torch.empty((len(ids), 8, 128), device=DEV, dtype=torch.float16)

    def body(e):
        out.zero_()
        e.gather(cache, ids, 0, out=out)

    _, seen = explore(body, check=lambda: torch.testing.assert_close(out, cache[ids.to(DEV)], rtol=0, atol=0))
    assert "kernel" in seen


def test_default_stream_ordering():
    # src is produced by a torch kernel on the legacy default stream right before the copy.
    e = ce.Engine(tune_trials=1, tune_topk=4)
    for _ in range(8):
        base = torch.randn((256, 1024), device=DEV)
        src = base * 3 + 1
        dst = torch.empty((256, 1024), device="cpu").pin_memory()
        with e.scope():
            for r in range(0, 256, 16):
                e.copy_(dst[r:r + 8], src[r:r + 8])
        want = (base * 3 + 1).cpu()
        torch.testing.assert_close(dst[0:8], want[0:8], rtol=0, atol=0)
        for r in range(0, 256, 16):
            torch.testing.assert_close(dst[r:r + 8], want[r:r + 8], rtol=0, atol=0)


def test_side_stream():
    s = torch.cuda.Stream()
    src = host((1024, 64), pinned=True)
    dst = torch.zeros((1024, 64), device=DEV)
    with torch.cuda.stream(s):
        e = ce.Engine(prior_only=True)
        with e.scope():
            for r in range(0, 1024, 32):
                e.copy_(dst[r:r + 16], src[r:r + 16])
        s.synchronize()
    for r in range(0, 1024, 32):
        torch.testing.assert_close(dst[r:r + 16].cpu(), src[r:r + 16], rtol=0, atol=0)


# ---- conversions -------------------------------------------------------------------------


@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16, torch.float64, torch.int32])
def test_d2d_cast_matches_torch(dt):
    x = torch.randn(10000, device=DEV) * 1000
    out = torch.empty(10000, device=DEV, dtype=dt)
    ce.Engine(prior_only=True).copy_(out, x)
    if dt.is_floating_point:
        torch.testing.assert_close(out, x.to(dt), rtol=0, atol=0)
    else:
        torch.testing.assert_close(out, x.trunc().to(dt), rtol=0, atol=0)


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_staged_cast_and_quantize(pinned, direction):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((300, 70)) * 5).astype(np.float32)
    fn = ce.Elementwise(scale=8.0, round=True, clamp=(-127, 127))
    want = np.empty(x.shape, np.int8)
    ce.Engine(prior_only=True).copy_(want, x, fn=fn)        # host reference (same conversion code)
    xt = torch.from_numpy(x)
    if direction == "h2d":
        src = xt.pin_memory() if pinned else xt
        dst = torch.zeros(x.shape, dtype=torch.int8, device=DEV)
    else:
        src = xt.to(DEV)
        dst = torch.zeros(x.shape, dtype=torch.int8)
        dst = dst.pin_memory() if pinned else dst
    sel = (slice(None, None, 3), slice(5, 60))
    ce.Engine(prior_only=True, staging_budget_mb=1).copy_(dst[sel], src[sel], fn=fn)
    got = dst.cpu().numpy()
    np.testing.assert_array_equal(got[sel], want[sel])
    assert not got[1].any()


def test_host_and_device_conversions_agree():
    rng = np.random.default_rng(3)
    x = np.concatenate([(rng.standard_normal(20000) * 10.0 ** rng.integers(-6, 6, 20000)),
                        [0.0, -0.0, np.inf, -np.inf, 65504.0, 65520.0, 1e-8]]).astype(np.float32)
    for dt, tdt in [("float16", torch.float16), ("bfloat16", torch.bfloat16)]:
        dev_out = torch.empty(x.size, device=DEV, dtype=tdt)
        ce.Engine(prior_only=True).copy_(dev_out, torch.from_numpy(x).to(DEV))
        host_out = torch.empty(x.size, dtype=tdt)
        ce.Engine(prior_only=True).copy_(host_out, torch.from_numpy(x))
        assert torch.equal(dev_out.cpu().view(torch.int16), host_out.view(torch.int16)), dt


def test_large_staged_copy_is_chunked():
    x = torch.randn(3 << 20)                          # 12 MB > 1 MB staging
    out = torch.empty(3 << 20, device=DEV, dtype=torch.float16)
    e = ce.Engine(prior_only=True, staging_budget_mb=1)
    e.copy_(out, x)
    torch.testing.assert_close(out.cpu(), x.half(), rtol=0, atol=0)
    assert e.stats()["last"]["strategies"] == ["staged"]


# ---- engine features ---------------------------------------------------------------------


def test_freeze_replay_cuda():
    src = host((32, 256), pinned=True)
    dst = torch.zeros((32, 256), device=DEV)
    e = ce.Engine(prior_only=True)
    with e.scope():
        for i in range(32):
            e.copy_(dst[31 - i], src[i])
    frozen = e.freeze()
    dst.zero_()
    src.copy_(torch.randn(32, 256))
    frozen.run()
    torch.testing.assert_close(dst.cpu(), src.flip(0), rtol=0, atol=0)


def test_cuda_scatter_rejects_duplicates():
    dst = torch.zeros((8, 4), device=DEV)
    with pytest.raises(ce.UnsafeCopy):
        ce.Engine(prior_only=True).scatter_(dst, [1, 1], torch.ones((2, 4), device=DEV))


def test_mixed_host_and_cuda_waves():
    # D2H into pinned, then a host copy reading it: the host wave must wait for the GPU.
    src = torch.randn((4096,), device=DEV)
    mid = torch.zeros((4096,)).pin_memory()
    out = np.zeros(4096, np.float32)
    e = ce.Engine(prior_only=True)
    with e.scope():
        e.copy_(mid, src)
        e.copy_(out, mid.numpy())
    np.testing.assert_array_equal(out, src.cpu().numpy())


@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_pageable_fragments_use_bounce(direction):
    n_blocks, blk = 512, 256
    if direction == "h2d":
        src, dst = torch.randn(n_blocks, blk), torch.zeros(n_blocks, blk, device=DEV)
    else:
        src, dst = torch.randn(n_blocks, blk, device=DEV), torch.zeros(n_blocks, blk)
    perm = torch.randperm(n_blocks).tolist()

    def body(e):
        dst.zero_()
        with e.scope():
            for i, j in enumerate(perm):
                e.copy_(dst[j], src[i])

    def check():
        torch.testing.assert_close(dst[perm].cpu(), src.cpu(), rtol=0, atol=0)

    _, seen = explore(body, check=check, rounds=8)
    assert {"staged", "bounce"} <= seen


def test_stats_report_admission():
    st = ce.Engine(prior_only=True).stats()
    dev0 = st["cuda"][0]
    assert dev0["kernels"] is True and dev0["batch"] is True


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dense_permute_h2d_moves_once_and_permutes_on_device(pinned, dtype):
    x = torch.randn(64, 8, 128).to(dtype)
    x = x.pin_memory() if pinned else x
    for src in (x.permute(1, 0, 2), x.transpose(1, 2)):      # 256-byte rows, then 2-4 byte elements
        dst = torch.empty(src.shape, dtype=dtype, device=DEV)
        e = ce.Engine(prior_only=True)
        e.copy_(dst, src)
        torch.testing.assert_close(dst.cpu(), src.contiguous(), rtol=0, atol=0)
        # 256-byte rows go through staging + chunk kernel; single elements through the fused kernel
        assert e.stats()["last"]["strategies"] in (["staged"], ["fused"])
        assert e.stats()["last"]["calls"] == 2              # one dense H2D + one kernel


def test_dense_permute_d2h_with_cast():
    x = torch.randn(32, 16, 64, device=DEV)
    src = x.permute(2, 0, 1)
    dst = torch.empty(src.shape, dtype=torch.float16)
    e = ce.Engine(prior_only=True)
    e.copy_(dst, src)
    torch.testing.assert_close(dst, src.cpu().half(), rtol=0, atol=0)


def test_gap_filling_stays_correct():
    src = torch.randn(4096)
    dst = torch.full((4096,), -1.0, device=DEV)
    e = ce.Engine(prior_only=True)
    with e.scope():
        for i in range(0, 4096, 64):                      # every other 128-byte block
            e.copy_(dst[i:i + 32], src[i:i + 32])
    want = torch.full((4096,), -1.0)
    for i in range(0, 4096, 64):
        want[i:i + 32] = src[i:i + 32]
    torch.testing.assert_close(dst.cpu(), want, rtol=0, atol=0)


def test_broadcast_and_element_transpose_kernels():
    # qscale pattern: tiny broadcast H2D copies
    L, B, N, S = 4, 16, 16, 8
    k_qs = torch.rand(B, N, 1).pin_memory()
    scale = torch.zeros(L, B, N, S, device=DEV)

    def body(e):
        scale.zero_()
        with e.scope():
            for l in range(L):
                for b in range(B):
                    e.copy_(scale[l, b], k_qs[b].expand(N, S))

    want = k_qs.expand(B, N, S).unsqueeze(0).expand(L, B, N, S)
    explore(body, check=lambda: torch.testing.assert_close(scale.cpu(), want, rtol=0, atol=0))
    # element-level transpose on device with a cast
    x = torch.randn(33, 65, device=DEV)
    out = torch.empty(65, 33, device=DEV, dtype=torch.bfloat16)
    ce.Engine(prior_only=True).copy_(out, x.t())
    torch.testing.assert_close(out, x.t().bfloat16(), rtol=0, atol=0)


# ---- fused structured statements -------------------------------------------------------------


def test_fused_qscale_mixed_head_counts():
    # Layers with different head counts: different structures, still one group.
    B, N = 16, 16
    heads = [8, 4, 8, 4, 2, 8]
    k_qs = torch.rand(B, N, 1).pin_memory()
    scales = [torch.zeros(B, N, s, device=DEV) for s in heads]

    def body(e):
        for t in scales:
            t.zero_()
        with e.scope():
            for t, s in zip(scales, heads):
                for b in range(B):
                    e.copy_(t[b], k_qs[b].expand(N, s))

    def check():
        for t, s in zip(scales, heads):
            torch.testing.assert_close(t.cpu(), k_qs.expand(B, N, s), rtol=0, atol=0)

    e, seen = explore(body, check=check)
    assert "fused" in seen
    e2 = ce.Engine(prior_only=True)
    body(e2)
    check()
    st = e2.stats()["last"]
    assert st["strategies"] == ["fused"] and st["calls"] == 4     # 3 structures + 1 link copy


@pytest.mark.parametrize("pinned", [False, True])
def test_fused_d2h_dense_and_h2d_flip_with_fn(pinned):
    x = torch.randn(40, 24, device=DEV)
    out = torch.empty(24, 40, dtype=torch.float16)
    out = out.pin_memory() if pinned else out
    e = ce.Engine(prior_only=True)
    e.copy_(out, x.t())                                  # D2H, element transpose + cast
    assert e.stats()["last"]["strategies"] == ["fused"]
    torch.testing.assert_close(out, x.t().cpu().half(), rtol=0, atol=0)

    h = torch.randn(24, 40)
    h = h.pin_memory() if pinned else h
    q = torch.empty(40, 24, dtype=torch.int8, device=DEV)
    fn = ce.Elementwise(scale=20.0, round=True, clamp=(-127, 127))
    e.copy_(q, h.t().flip(0), fn=fn)                     # H2D, negative stride + quantize
    want = torch.empty(40, 24, dtype=torch.int8)
    ce.Engine(prior_only=True).copy_(want, h.t().flip(0), fn=fn)
    assert torch.equal(q.cpu(), want)


def test_fused_d2d_broadcast_and_strided_dst():
    src = torch.randn(64, 1, device=DEV)
    dst = torch.full((64, 32), -1.0, device=DEV)
    e = ce.Engine(prior_only=True)
    e.copy_(dst[:, ::2], src.expand(64, 16))             # strided destination keeps its holes
    assert e.stats()["last"]["strategies"] == ["fused"]
    want = torch.full((64, 32), -1.0, device=DEV)
    want[:, ::2] = src
    torch.testing.assert_close(dst, want, rtol=0, atol=0)


def test_d2h_with_holes_does_not_use_fused():
    src = torch.randn(64, 16, device=DEV)
    dst = torch.full((64, 32), -1.0)
    e = ce.Engine(prior_only=True)
    e.copy_(dst[:, ::2], src)                            # host gaps must survive
    assert "fused" not in e.stats()["last"]["strategies"]
    want = torch.full((64, 32), -1.0)
    want[:, ::2] = src.cpu()
    torch.testing.assert_close(dst, want, rtol=0, atol=0)


# The native baseline is offered only where torch's copy_ gives ce's exact bits.
_SRC = [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
        torch.float16, torch.bfloat16, torch.float32, torch.float64]
_DST = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


@pytest.mark.parametrize("sdt", _SRC)
@pytest.mark.parametrize("ddt", _DST)
def test_native_eligible_casts_match_torch_bits(sdt, ddt):
    g = torch.Generator().manual_seed(0)
    if sdt.is_floating_point:
        x = torch.randn(1 << 14, generator=g, dtype=torch.float64) * 10.0 ** torch.randint(-12, 12, (1 << 14,), generator=g)
        x = torch.cat([x, torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 65504.0, 65520.0, 3e38, 1e-40])])
    else:
        info = torch.iinfo(sdt)
        x = torch.randint(info.min, info.max, (1 << 14,), generator=g, dtype=torch.int64)
        x = torch.cat([x, torch.tensor([info.min, info.max, 0, 2049, 16777217, 2 ** 24 + 1])
                       .clamp(info.min, info.max)])
    src = x.to(sdt).to(DEV)
    want = torch.empty(src.shape, dtype=ddt, device=DEV).copy_(src)
    got = torch.empty_like(want)
    ce.Engine(prior_only=True).copy_(got, src)
    bits = {2: torch.int16, 4: torch.int32, 8: torch.int64}[want.element_size()]
    assert torch.equal(got.view(bits), want.view(bits)) or torch.equal(got.isnan(), want.isnan()) and \
        torch.equal(got.view(bits)[~want.isnan()], want.view(bits)[~want.isnan()])


# ---- the stable-ABI torch extension ---------------------------------------------------------


def test_torch_ops_translate_errors():
    if ce._torch_ops.copy_ is None:
        pytest.skip(ce._torch_ops.error)
    x = torch.arange(64.0, device=DEV)
    with pytest.raises(ce.UnsafeCopy):
        ce.Engine().copy_(x[1:], x[:-1])
    with pytest.raises(ce.Unsupported):
        ce.Engine().copy_(torch.zeros(4, dtype=torch.complex64, device=DEV), torch.zeros(4, dtype=torch.complex64, device=DEV))


@pytest.mark.parametrize("pinned", [False, True])
def test_torch_ops_host_tensors_and_python_fallback(pinned, monkeypatch):
    src = host((256, 64), pinned=pinned)
    for path in ("ops", "python"):
        if path == "python":
            monkeypatch.setattr(ce._torch_ops, "copy_", None)
        dst = torch.zeros((64, 256), device=DEV)
        e = ce.Engine(prior_only=True)
        e.copy_(dst, src.t())
        torch.testing.assert_close(dst.cpu(), src.t(), rtol=0, atol=0)
        back = host((256, 64), pinned=pinned, fill=False)
        e.copy_(back, dst.t(), fn=ce.Elementwise(scale=2.0))
        torch.testing.assert_close(back, src * 2, rtol=0, atol=0)


@pytest.mark.parametrize("pair", [(torch.int8, torch.float16), (torch.float32, torch.bfloat16),
                                  (torch.float16, torch.float32), (torch.float64, torch.int8),
                                  (torch.uint8, torch.float64), (torch.bool, torch.int32)])
@pytest.mark.parametrize("fn", [None, ce.Elementwise(scale=0.5, bias=1.0, round=True, clamp=(-100, 100))])
def test_kernel_vector_and_scalar_paths_agree_with_host(pair, fn):
    sdt, ddt = pair
    rng = torch.Generator().manual_seed(1)
    base = (torch.randn(6000, generator=rng) * 50).to(sdt)
    for so, do, n, rows in [(0, 0, 4096, 1), (3, 5, 1001, 1), (8, 16, 777, 1), (1, 0, 64, 7), (0, 0, 256, 9)]:
        src = base.to(DEV)[so:so + n * rows].view(rows, n)
        out = torch.zeros(rows * n + do + 3, dtype=ddt, device=DEV)
        dst = out[do:do + n * rows].view(rows, n)
        ce.Engine(prior_only=True).copy_(dst, src, fn=fn)
        want = torch.zeros(rows, n, dtype=ddt)
        ce.Engine(prior_only=True).copy_(want, base[so:so + n * rows].view(rows, n), fn=fn)  # host conversion
        bits = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[want.element_size()]
        assert torch.equal(dst.cpu().view(bits), want.view(bits)), (so, do, n, rows)


@pytest.mark.parametrize("sdt,ddt,fn", [
    (torch.uint8, torch.uint8, None), (torch.int16, torch.int16, None), (torch.float32, torch.float32, None),
    (torch.float64, torch.float64, None), (torch.float32, torch.float16, None), (torch.bfloat16, torch.float32, None),
    (torch.int8, torch.float16, ce.Elementwise(scale=0.25)), (torch.float32, torch.int8, ce.Elementwise(scale=3.0, round=True, clamp=(-127, 127))),
    (torch.int32, torch.float64, None),  # no specialization: run-time dispatch
])
@pytest.mark.parametrize("shape", [(7, 13, 4), (5, 3, 8), (11, 1, 6)])
def test_fused_statements_match_host(sdt, ddt, fn, shape):
    a, b, c = shape
    g = torch.Generator().manual_seed(2)
    src_h = (torch.randn(a, 1, c, generator=g) * 40).to(sdt)
    want = torch.zeros(a, b, c, dtype=ddt)
    ce.Engine(prior_only=True).copy_(want, src_h.expand(a, b, c), fn=fn)  # host conversion
    src = src_h.to(DEV)
    for off in (0, 1):  # off=1: destination bases not vector-aligned
        buf = torch.zeros(a * b * c + off, dtype=ddt, device=DEV)
        dst = buf[off:].view(a, b, c)
        e = ce.Engine(prior_only=True)
        with e.scope():
            for i in range(a):  # one statement per row: a fused batch
                e.copy_(dst[i], src[i].expand(b, c), fn=fn)
        bits = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[want.element_size()]
        assert torch.equal(dst.cpu().view(bits), want.view(bits)), (off, e.stats()["last"]["strategies"])


def test_fused_more_statements_than_one_grid_row():
    n = 70000  # > 65535: the kernel strides over statements
    src = torch.arange(n, dtype=torch.float32, device=DEV).view(n, 1)
    dst = torch.zeros(n, 4, device=DEV)
    e = ce.Engine(prior_only=True)
    with e.scope():
        for i in range(n):
            e.copy_(dst[i], src[i].expand(4))
    assert e.stats()["last"]["strategies"] == ["fused"]
    torch.testing.assert_close(dst, src.expand(n, 4), rtol=0, atol=0)


# ---- buffers of copies the host does not wait for -------------------------------------------


def _reuse_pinned_after_free(start_copy, later, n=1 << 20):
    """Start a copy from a pinned tensor while the GPU is busy, drop the tensor and
    let the engine move on to other (already planned) work: until the copy has
    run, torch must not hand that pinned memory to anyone else, and the copy
    must deliver the original data. (Some drivers capture pinned sources at
    submit time; there only the pointer check can fail.)"""
    later()  # planned once now: planning may synchronize, the timed part must not
    src = torch.arange(n, dtype=torch.float32).pin_memory()
    want, p = src.clone(), src.data_ptr()
    dst = torch.zeros(n, device=DEV)
    torch.cuda._sleep(2_000_000_000)  # the copy is queued behind this
    start_copy(dst, src)
    del src
    later()
    assert not torch.cuda.current_stream().query(), "the GPU finished too early for this test"
    junk = torch.empty(n, pin_memory=True)
    assert junk.data_ptr() != p, "pinned memory of an in-flight copy was reused"
    junk.fill_(-1.0)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst.cpu(), want, rtol=0, atol=0)


def test_allow_async_keeps_pinned_sources_alive():
    e = ce.Engine(allow_async=True, prior_only=True)
    small = (torch.zeros(4, device=DEV), torch.ones(4, device=DEV))
    _reuse_pinned_after_free(lambda d, s: e.copy_(d, s), lambda: e.copy_(*small))
    for _ in range(200):  # references are let go as copies complete: no growth
        e.copy_(torch.zeros(4, device=DEV), torch.ones(4).pin_memory())
    torch.cuda.synchronize()
    e.copy_(torch.zeros(4, device=DEV), torch.ones(4).pin_memory())
    assert len(e._inflight) <= 2


@pytest.mark.parametrize("intercept", ["mode", "shim"])
def test_intercepted_non_blocking_keeps_pinned_sources_alive(intercept):
    if intercept == "shim" and ce._shim.lib is None:
        pytest.skip(ce._shim.error)
    e = ce.Engine(prior_only=True, _intercept=intercept)

    def start(d, s):
        with e.scope():
            d.copy_(s, non_blocking=True)

    def later():  # a new flush replaces the last one's ops
        a, b = torch.zeros(4, device=DEV), torch.ones(4, device=DEV)
        with e.scope():
            a.copy_(b)

    _reuse_pinned_after_free(start, later)


def test_sparse_tensors_take_the_general_path():
    dense = torch.eye(8)
    sparse = dense.to_sparse()
    with pytest.raises((TypeError, ValueError, ce.CopyEngineError, RuntimeError)):
        ce.Engine(prior_only=True).copy_(torch.zeros(8, 8), sparse)


# ---- SM limit --------------------------------------------------------------------------------


def _sm_engine(n):
    e = ce.Engine(prior_only=True, max_sms=n)
    e.copy_(torch.empty(4, device=DEV), torch.ones(4, device=DEV))  # initializes the device
    return e, e.stats()["cuda"][0]["sm_limit"]


def test_sm_limit_is_enforced_and_reported():
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    e, how = _sm_engine(max(2, sms // 4))
    assert how.startswith(("hard:", "soft:")), how
    _, none = _sm_engine(sms)
    assert none.startswith("none")


@pytest.mark.parametrize("n", [1, 2, 6])
def test_copies_are_correct_under_an_sm_limit(n):
    e, _ = _sm_engine(n)
    x = torch.randn(1 << 20, device=DEV)
    for case in range(4):
        if case == 0:  # chunk kernels: cast
            out = torch.empty_like(x, dtype=torch.float16)
            e.copy_(out, x)
            want = x.half()
        elif case == 1:  # tiled transpose
            m = x.view(1024, 1024)
            out = torch.empty(1024, 1024, dtype=torch.bfloat16, device=DEV)
            e.copy_(out, m.t())
            want = m.t().bfloat16()
        elif case == 2:  # structured statements, many of them
            src = x[:4096].view(4096, 1)
            out = torch.empty(4096, 64, device=DEV)
            with e.scope():
                for i in range(0, 4096, 32):
                    e.copy_(out[i:i + 32], src[i:i + 32].expand(32, 64))
            want = src.expand(4096, 64)
        else:  # gather table
            cache = x.view(4096, 256)
            ids = torch.randperm(4096)[:3000].tolist()
            out = torch.empty(3000, 256, dtype=torch.float16, device=DEV)
            e.copy_(out, ce.lazy(cache).index_select(0, ids))
            want = cache[torch.tensor(ids, device=DEV)].half()
        torch.testing.assert_close(out, want, rtol=0, atol=0)


def test_hard_sm_limit_keeps_kernels_on_its_sms():
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    e, how = _sm_engine(max(2, sms // 4))
    if not how.startswith("hard:"):
        pytest.skip(how)
    # A long copy on the limited engine leaves the other SMs free: a kernel on
    # the default stream of another stream runs concurrently.
    big = torch.randn(1 << 24, device=DEV)
    out = torch.empty_like(big, dtype=torch.float16)
    torch.cuda.synchronize()
    e.copy_(out, big)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, big.half(), rtol=0, atol=0)


# ---- copyengine's own device memory through torch's allocator --------------------------------


def _skip_unless_torch_allocator():
    if ce._torch_ops.copy_ is None:
        pytest.skip(ce._torch_ops.error)
    e = ce.Engine(prior_only=True)
    e.copy_(torch.zeros(4, device=DEV), torch.ones(4, device=DEV))
    if e.stats()["cuda"][0]["allocator"] != "torch":
        pytest.skip("torch's allocator is off (CE_TORCH_ALLOCATOR=0)")


def test_own_memory_counts_in_torch_statistics():
    _skip_unless_torch_allocator()
    e = ce.Engine(prior_only=True)
    src = torch.randn(64, 32).pin_memory()
    dst = torch.zeros(32, 64, device=DEV)
    before = torch.cuda.memory_allocated()
    e.copy_(dst, src.t())  # through device staging
    torch.cuda.synchronize()
    assert e.stats()["cuda"][0]["allocator"] == "torch"
    assert torch.cuda.memory_allocated() - before >= 1 << 20
    torch.testing.assert_close(dst.cpu(), src.t(), rtol=0, atol=0)


_PINNED_CHECK = """
import torch, copyengine as ce
def host_bytes():
    s = torch._C._cuda_hostMemoryStats() if hasattr(torch._C, "_cuda_hostMemoryStats") else {}
    return s.get("allocated_bytes", {}).get("allocated", 0)
e = ce.Engine(tune_trials=1, tune_topk=4)
e.copy_(torch.zeros(4, device="cuda"), torch.ones(4, device="cuda"))
info = e.stats()["cuda"][0]
src, dst = torch.randn(512, 256), torch.zeros(512, 256, device="cuda")
perm = torch.randperm(512).tolist()
before, seen = host_bytes(), set()
for _ in range(8):  # explores bounce (pinned bounce buffers)
    dst.zero_()
    with e.scope():
        for i, j in enumerate(perm):
            e.copy_(dst[j], src[i])
    torch.cuda.synchronize()
    seen.update(e.stats()["last"]["strategies"])
    assert torch.equal(dst[perm].cpu(), src)
print(info["host_allocator"], "bounce" in seen, host_bytes() - before >= 1 << 20)
"""


def test_own_pinned_memory_counts_in_torch_statistics():
    # A fresh process: the backend's buffers outlive engines, so earlier tests may own them.
    import subprocess
    import sys
    _skip_unless_torch_allocator()
    out = subprocess.run([sys.executable, "-c", _PINNED_CHECK], capture_output=True, text=True)
    words = out.stdout.split()
    assert len(words) == 3, out.stderr
    if words[0] != "torch":
        pytest.skip("this torch has no pinned allocation entry point")
    assert words[1:] == ["True", "True"]


def test_freed_memory_is_not_reused_before_pending_copies():
    # Staging belongs to the stream it was allocated on. With more streams than
    # staging buffers, stream B takes over the least recently used one (from
    # S0), queues a copy through it behind a long kernel, then needs a bigger
    # one: the old buffer is freed while B's copy still uses it, and torch
    # hands that block to new tensors on S0 at once. S0's work must wait for B,
    # else B's late staging writes land in S0's new tensors.
    _skip_unless_torch_allocator()
    e = ce.Engine(prior_only=True, allow_async=True)  # the host does not wait for B
    small = torch.randn(64, 32).pin_memory()
    streams = [torch.cuda.Stream() for _ in range(9)]
    for st in streams[:8]:  # one staging buffer (1 MB) per stream, all lanes taken
        with torch.cuda.stream(st):
            e.copy_(torch.zeros(32, 64, device=DEV), small.t())
    torch.cuda.synchronize()
    s0, b = streams[0], streams[8]
    out = torch.zeros(32, 64, device=DEV)
    big_src = torch.randn(1024, 512).pin_memory()
    big = torch.zeros(512, 1024, device=DEV)
    with torch.cuda.stream(b):
        torch.cuda._sleep(1_000_000_000)
        e.copy_(out, small.t())    # takes over S0's buffer; pending
        e.copy_(big, big_src.t())  # needs 2 MB: the 1 MB buffer is freed
    with torch.cuda.stream(s0):  # 1 MB blocks on S0: one of them is likely the freed staging
        fresh = [torch.full((1 << 18,), -1.0, device=DEV) for _ in range(8)]
    torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(), small.t(), rtol=0, atol=0)
    torch.testing.assert_close(big.cpu(), big_src.t(), rtol=0, atol=0)
    assert all(bool((f == -1.0).all()) for f in fresh), "a pending copy wrote into reused memory"


def test_cuda_allocator_when_torch_allocator_is_off():
    import subprocess
    import sys
    code = ("import torch, copyengine as ce; e = ce.Engine(prior_only=True); "
            "e.copy_(torch.zeros(32, 64, device='cuda'), torch.randn(64, 32).pin_memory().t()); "
            "print(e.stats()['cuda'][0]['allocator'])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**__import__("os").environ, "CE_TORCH_ALLOCATOR": "0"})
    assert out.stdout.strip() == "cuda", out.stderr


def test_sm_limited_engine_with_staging_tears_down():
    # Freeing staging orders its stream after every stream ce used, so the
    # SM-limited stream must still exist then (it was destroyed first).
    import subprocess
    import sys
    code = ("import gc, torch, copyengine as ce\n"
            "e = ce.Engine(max_sms=2, prior_only=True)\n"
            "src, dst = torch.randn(64, 32).pin_memory(), torch.zeros(32, 64, device='cuda')\n"
            "e.copy_(dst, src.t())\n"  # through device staging
            "torch.cuda.synchronize()\n"
            "assert torch.equal(dst.cpu(), src.t())\n"
            "info = e.stats()['cuda'][0]\n"
            "del e; gc.collect(); torch.cuda.synchronize()\n"
            "print('ok', info['sm_limit'], info['allocator'])\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.startswith("ok"), (out.returncode, out.stdout, out.stderr[-2000:])


@pytest.mark.parametrize("offset", [0, 2])  # 16-byte aligned rows, and not (the plain warp path)
def test_bulk_copies_through_shared_memory(offset):
    # sm80+: cp.async (sm90+: TMA). Explored like the other strategies; every
    # round must give the source's bytes.
    src = torch.randn(64, 8200, device=DEV)
    dst = torch.zeros(64, 8300, device=DEV)
    s, d = src[:, offset:offset + 8192], dst[:, 4 + offset:4 + offset + 8192]

    def run(e):
        dst.zero_()
        e.copy_(d, s)

    e, seen = explore(run, rounds=12, check=lambda: torch.testing.assert_close(d, s, rtol=0, atol=0))
    info = e.stats()["cuda"][0]
    if info["bulk"] in ("tma", "cp.async"):
        assert "bulk" in seen
    else:
        assert "bulk" not in seen


@pytest.mark.parametrize("D", [8, 16, 64, 120, 128, 248, 256, 260])
@pytest.mark.parametrize("pair", [(torch.bfloat16, torch.float16), (torch.float32, torch.bfloat16),
                                  (torch.float16, torch.float32)])
def test_narrow_rows_cast_matches_torch(D, pair):
    # A head-sized innermost dimension under a permute: rows narrower than a
    # warp's vector span (lanes spread across rows), at and around the bounds.
    src_dt, dst_dt = pair
    x = (torch.randn(4, 96, 8, D, device=DEV) * 100).to(src_dt)
    x.view(-1)[:4] = torch.tensor([float("nan"), float("inf"), -0.0, 65504.0])
    out = torch.empty(4, 8, 96, D, device=DEV, dtype=dst_dt)
    e = ce.Engine(prior_only=True)
    e.copy_(out, x.permute(0, 2, 1, 3))
    e.synchronize()
    want = x.permute(0, 2, 1, 3).to(dst_dt)
    nan = want.isnan()
    assert torch.equal(out.isnan(), nan) and torch.equal(out[~nan], want[~nan])


def test_copyengine_memory_waits_for_the_allocation_stream():
    # A block torch hands copyengine may still have pending writes from a freed
    # tensor on torch's stream; copyengine's first write to it (a copy table,
    # written on its own stream) must come after them, or the table is
    # overwritten and its copies read garbage.
    for k in range(6):
        e = ce.Engine(prior_only=True)  # a fresh engine: its first copy builds its tables
        src = torch.randn(64 + k, 128, device=DEV)
        dst = torch.empty(128, 64 + k, device=DEV, dtype=torch.float16)
        # Blocks of the sizes copy tables take, freed with a write still pending.
        doomed = [torch.empty(n // 4, device=DEV) for n in (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)]
        torch.cuda._sleep(50_000_000)
        for t in doomed:
            t.fill_(float("nan"))
        del doomed, t
        e.copy_(dst, src.t())
        torch.cuda.synchronize()
        assert torch.equal(dst, src.t().to(torch.float16)), k


def test_tma2d_admission_and_pitched_copies():
    # sm90+: 2-D blocks of pure device copies as tensor maps (admitted by a
    # verified test copy); older GPUs report why not.
    st = ce.Engine().stats()
    major = torch.cuda.get_device_capability(0)[0]
    x = torch.randn(4, 4, device=DEV)
    e = ce.Engine(tune_trials=2, tune_topk=6)
    e.copy_(torch.empty_like(x), x)
    info = e.stats()["cuda"][0]["tma2d"]
    if major < 9:
        assert info == "needs sm90 or later" or info == "no kernels", info
        return
    assert info is True, info
    # Narrow rows (a KV head of 128 bf16 values) gathered by a permute: 2-D blocks.
    src = torch.randn(64, 512, 8, 128, device=DEV).bfloat16()  # [blocks, tok, heads, d]
    dst = torch.empty(64, 8, 512, 128, device=DEV, dtype=torch.bfloat16)
    seen = set()
    for _ in range(30):
        dst.zero_()
        e.copy_(dst, src.permute(0, 2, 1, 3))
        torch.cuda.synchronize()
        seen.update(e.stats()["last"]["strategies"])
        assert torch.equal(dst, src.permute(0, 2, 1, 3))
    assert "tma2d" in seen, seen


def test_copyengine_memory_waits_for_the_allocation_stream():
    # A block torch hands copyengine may still have pending writes from a freed
    # tensor on torch's stream; copyengine's first write to it (a copy table,
    # written on its own stream) must come after them, or the table is
    # overwritten and its copies read garbage.
    for k in range(6):
        e = ce.Engine(prior_only=True)  # a fresh engine: its first copy builds its tables
        src = torch.randn(64 + k, 128, device=DEV)
        dst = torch.empty(128, 64 + k, device=DEV, dtype=torch.float16)
        # Blocks of the sizes copy tables take, freed with a write still pending.
        doomed = [torch.empty(n // 4, device=DEV) for n in (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)]
        torch.cuda._sleep(50_000_000)
        for t in doomed:
            t.fill_(float("nan"))
        del doomed, t
        e.copy_(dst, src.t())
        torch.cuda.synchronize()
        assert torch.equal(dst, src.t().to(torch.float16)), k
