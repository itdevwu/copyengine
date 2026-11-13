"""CUDA paths, checked against torch. Skipped without torch or a GPU."""

import numpy as np
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
