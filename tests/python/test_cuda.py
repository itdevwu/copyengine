"""CUDA paths, checked against torch. Skipped without torch or a GPU."""

import numpy as np
import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

DEV = "cuda"


def explore(fn, rounds=8, check=None):
    """Run fn(engine) with exploration so every candidate strategy executes; check each round."""
    e = ce.Engine(tune=True, tune_trials=1, tune_topk=4)
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
    assert "passthrough" in seen
    back = host((16, 4, 32), pinned=pinned, fill=False)
    ce.Engine(tune=False).copy_(back, view_d)
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
    assert seen >= {"batch", "runs", "passthrough"}


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
    e = ce.Engine(tune=True, tune_trials=1, tune_topk=4)
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
        e = ce.Engine(tune=False)
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
    ce.Engine(tune=False).copy_(out, x)
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
    ce.Engine(tune=False).copy_(want, x, fn=fn)        # host reference (same conversion code)
    xt = torch.from_numpy(x)
    if direction == "h2d":
        src = xt.pin_memory() if pinned else xt
        dst = torch.zeros(x.shape, dtype=torch.int8, device=DEV)
    else:
        src = xt.to(DEV)
        dst = torch.zeros(x.shape, dtype=torch.int8)
        dst = dst.pin_memory() if pinned else dst
    sel = (slice(None, None, 3), slice(5, 60))
    ce.Engine(tune=False, staging_budget_mb=1).copy_(dst[sel], src[sel], fn=fn)
    got = dst.cpu().numpy()
    np.testing.assert_array_equal(got[sel], want[sel])
    assert not got[1].any()


def test_host_and_device_conversions_agree():
    rng = np.random.default_rng(3)
    x = np.concatenate([(rng.standard_normal(20000) * 10.0 ** rng.integers(-6, 6, 20000)),
                        [0.0, -0.0, np.inf, -np.inf, 65504.0, 65520.0, 1e-8]]).astype(np.float32)
    for dt, tdt in [("float16", torch.float16), ("bfloat16", torch.bfloat16)]:
        dev_out = torch.empty(x.size, device=DEV, dtype=tdt)
        ce.Engine(tune=False).copy_(dev_out, torch.from_numpy(x).to(DEV))
        host_out = torch.empty(x.size, dtype=tdt)
        ce.Engine(tune=False).copy_(host_out, torch.from_numpy(x))
        assert torch.equal(dev_out.cpu().view(torch.int16), host_out.view(torch.int16)), dt


def test_large_staged_copy_is_chunked():
    x = torch.randn(3 << 20)                          # 12 MB > 1 MB staging
    out = torch.empty(3 << 20, device=DEV, dtype=torch.float16)
    e = ce.Engine(tune=False, staging_budget_mb=1)
    e.copy_(out, x)
    torch.testing.assert_close(out.cpu(), x.half(), rtol=0, atol=0)
    assert e.stats()["last"]["strategies"] == ["staged"]


# ---- engine features ---------------------------------------------------------------------


def test_freeze_replay_cuda():
    src = host((32, 256), pinned=True)
    dst = torch.zeros((32, 256), device=DEV)
    e = ce.Engine(tune=False)
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
        ce.Engine(tune=False).scatter_(dst, [1, 1], torch.ones((2, 4), device=DEV))


def test_mixed_host_and_cuda_waves():
    # D2H into pinned, then a host copy reading it: the host wave must wait for the GPU.
    src = torch.randn((4096,), device=DEV)
    mid = torch.zeros((4096,)).pin_memory()
    out = np.zeros(4096, np.float32)
    e = ce.Engine(tune=False)
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
    st = ce.Engine(tune=False).stats()
    dev0 = st["cuda"][0]
    assert dev0["kernels"] is True and dev0["batch"] is True


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dense_permute_h2d_moves_once_and_permutes_on_device(pinned, dtype):
    x = torch.randn(64, 8, 128).to(dtype)
    x = x.pin_memory() if pinned else x
    for src in (x.permute(1, 0, 2), x.transpose(1, 2)):      # 256-byte rows, then 2-4 byte elements
        dst = torch.empty(src.shape, dtype=dtype, device=DEV)
        e = ce.Engine(tune=False)
        e.copy_(dst, src)
        torch.testing.assert_close(dst.cpu(), src.contiguous(), rtol=0, atol=0)
        assert e.stats()["last"]["strategies"] == ["staged"]
        assert e.stats()["last"]["calls"] == 2              # one dense H2D + one kernel


def test_dense_permute_d2h_with_cast():
    x = torch.randn(32, 16, 64, device=DEV)
    src = x.permute(2, 0, 1)
    dst = torch.empty(src.shape, dtype=torch.float16)
    e = ce.Engine(tune=False)
    e.copy_(dst, src)
    torch.testing.assert_close(dst, src.cpu().half(), rtol=0, atol=0)


def test_gap_filling_stays_correct():
    src = torch.randn(4096)
    dst = torch.full((4096,), -1.0, device=DEV)
    e = ce.Engine(tune=False)
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
    ce.Engine(tune=False).copy_(out, x.t())
    torch.testing.assert_close(out, x.t().bfloat16(), rtol=0, atol=0)
