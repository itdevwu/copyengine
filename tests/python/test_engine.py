import json
import time

import numpy as np
import pytest

import copyengine as ce


def engine(**kw):
    kw.setdefault("prior_only", True)
    return ce.Engine(**kw)


# ---- conversions: host semantics ---------------------------------------------------


def bf16_bits(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    r[nan] = ((u[nan] >> 16) | 0x40).astype(np.uint16)
    return r


def test_float16_matches_numpy_bits():
    rng = np.random.default_rng(0)
    x = np.concatenate([
        rng.standard_normal(5000).astype(np.float32) * 10.0 ** rng.integers(-8, 6, 5000),
        np.array([0.0, -0.0, np.inf, -np.inf, 65504.0, 65520.0, 6e-8, 3e-8, 1e-10], np.float32),
    ]).astype(np.float32)
    out = np.empty(x.shape, np.float16)
    engine().copy_(out, x)
    with np.errstate(over="ignore"):
        want = x.astype(np.float16)
    np.testing.assert_array_equal(out.view(np.uint16), want.view(np.uint16))
    back = np.empty(x.shape, np.float32)
    engine().copy_(back, out)
    np.testing.assert_array_equal(back, out.astype(np.float32))


def test_bfloat16_round_trip_via_uint16_view():
    # numpy has no bfloat16: write into a uint16 buffer viewed through segments with dtype names
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(4096) * 100).astype(np.float32)
    want = bf16_bits(x)
    out = np.zeros(x.size, np.uint16)
    e = engine()
    src = ce.Runs.of([0], [0], [x.size])
    sep = ce._core.Endpoint("host", 0, x.ctypes.data, x.nbytes, "float32")
    dep = ce._core.Endpoint("host", 0, out.ctypes.data, out.nbytes, "bfloat16")
    e._e.copy_runs(sep, dep, src.src_off, src.dst_off, src.length, ce.Elementwise())
    np.testing.assert_array_equal(out, want)


def test_float_to_int_saturates_and_truncates():
    x = np.array([-1e10, -129.7, -128.5, -0.9, 0.9, 126.9, 127.5, 1e10, np.nan, -np.inf, np.inf], np.float32)
    out = np.empty(x.shape, np.int8)
    engine().copy_(out, x)
    want = np.clip(np.nan_to_num(np.trunc(x.astype(np.float64)), nan=0.0, posinf=127, neginf=-128), -128, 127)
    np.testing.assert_array_equal(out, want.astype(np.int8))


def test_int_casts_and_bool():
    x = np.array([-300, -1, 0, 1, 255, 256, 70000], np.int64)
    out8 = np.empty(x.shape, np.uint8)
    engine().copy_(out8, x)
    np.testing.assert_array_equal(out8, x.astype(np.uint8))     # wraps
    outb = np.empty(x.shape, np.bool_)
    engine().copy_(outb, x)
    np.testing.assert_array_equal(outb, x != 0)


def test_affine_quantize_dequantize():
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(1000) * 3).astype(np.float32)
    q = np.empty(x.shape, np.int8)
    qs = np.float32(0.05)
    engine().copy_(q, x, fn=ce.Elementwise(scale=1 / qs, round=True, clamp=(-127, 127)))
    want = np.clip(np.rint(x * np.float32(1 / qs)), -127, 127).astype(np.int8)
    np.testing.assert_array_equal(q, want)
    deq = np.empty(x.shape, np.float16)
    engine().copy_(deq, q, fn=ce.Elementwise(scale=float(qs)))
    np.testing.assert_array_equal(deq, (q.astype(np.float32) * qs).astype(np.float16))


# ---- copy_ / gather / scatter ------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_copy_strided_views(seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((6, 5, 4)).astype(np.float32)
    b = np.zeros((5, 6, 4), np.float64)
    src = a.transpose(1, 0, 2)[:, ::-1]
    sl = (slice(None), slice(None), slice(None, None, int(rng.choice([1, 2]))))
    engine().copy_(b[sl], src[sl])
    want = np.zeros_like(b)
    want[sl] = src[sl]
    np.testing.assert_array_equal(b, want)


def test_broadcast_source():
    src = np.arange(4, dtype=np.float32)
    dst = np.zeros((3, 4), np.float32)
    engine().copy_(dst, src)
    np.testing.assert_array_equal(dst, np.broadcast_to(src, (3, 4)))


def test_gather_and_scatter():
    e = engine()
    cache = np.arange(10 * 8, dtype=np.float32).reshape(10, 8)
    ids = [7, 2, 2, 5]
    out = e.gather(cache, ids)
    np.testing.assert_array_equal(out, cache[ids])
    dst = np.zeros((10, 8), np.float16)
    e.scatter_(dst, [1, 4, 9], cache[:3])
    want = np.zeros((10, 8), np.float16)
    want[[1, 4, 9]] = cache[:3]
    np.testing.assert_array_equal(dst, want)
    with pytest.raises(ce.UnsafeCopy):
        e.scatter_(dst, [1, 1], cache[:2])


def test_overlap_is_rejected():
    a = np.arange(16, dtype=np.float32)
    with pytest.raises(ce.UnsafeCopy):
        engine().copy_(a[1:9], a[0:8])


def test_bounds_are_checked():
    a = np.zeros(8, np.float32)
    b = np.zeros(8, np.float32)
    with pytest.raises(ValueError):
        engine().segments(a, b, [4], [0], [5])
    with pytest.raises(ValueError):
        engine().segments(a, b, [0], [7], [2])


def test_valid_src_hint():
    src = np.arange(32, dtype=np.float32)
    dst = np.full(32, -1, np.float32)
    engine().segments(src, dst, [0, 16], [0, 16], [16, 16], valid_src=([0, 16], [5, 9]))
    want = np.full(32, -1, np.float32)
    want[0:5] = src[0:5]
    want[16:25] = src[16:25]
    np.testing.assert_array_equal(dst, want)


# ---- merging, plans, freeze --------------------------------------------------------------


def test_layer_loop_merges_into_one_call():
    L, n, pitch = 12, 64, 100
    slot = np.arange(L * n, dtype=np.float32)
    dst = np.zeros(L * pitch, np.float32)
    e = engine()
    with e.scope():
        for i in range(L):
            e.segments(slot, dst, [i * n], [i * pitch], [n])
    st = e.stats()["last"]
    assert st["ops"] == L and st["calls"] == 1 and st["waves"] == 1
    for i in range(L):
        np.testing.assert_array_equal(dst[i * pitch:i * pitch + n], slot[i * n:(i + 1) * n])


def test_separate_buffers_merge_by_address():
    # Views of one allocation passed as separate arrays still merge.
    big = np.arange(4 * 256, dtype=np.float32)
    out = np.zeros(4 * 256, np.float32)
    e = engine()
    with e.scope():
        for i in range(4):
            e.copy_(out[i * 256:(i + 1) * 256], big[i * 256:(i + 1) * 256])
    assert e.stats()["last"]["calls"] == 1
    np.testing.assert_array_equal(out, big)


def test_plan_cache_and_freeze():
    src = np.arange(64, dtype=np.float32)
    dst = np.zeros(64, np.float32)
    e = engine()

    def body():
        with e.scope():
            for i in range(8):
                e.segments(src, dst, [i * 8], [56 - i * 8], [8])

    body()
    assert e.stats()["last"]["plan_cache"] == "miss"
    body()
    assert e.stats()["last"]["plan_cache"] == "hit"
    frozen = e.freeze()
    dst[:] = 0
    src[:] = src[::-1].copy()
    frozen.run()
    np.testing.assert_array_equal(dst, np.concatenate([src[i * 8:(i + 1) * 8] for i in range(7, -1, -1)]))


def apply_reference(bufs, prog):
    for (s, d, so, do, n) in prog:
        bufs[d][do:do + n] = bufs[s][so:so + n].copy()


@pytest.mark.parametrize("seed", range(200))
def test_random_programs_lazy_equals_eager(seed):
    rng = np.random.default_rng(seed)
    nb = 3
    size = 64
    init = [rng.standard_normal(size).astype(np.float32) for _ in range(nb)]
    prog = []
    for _ in range(int(rng.integers(1, 25))):
        s, d = int(rng.integers(nb)), int(rng.integers(nb))
        n = int(rng.integers(1, 9))
        so, do = int(rng.integers(0, size - n + 1)), int(rng.integers(0, size - n + 1))
        if s == d and abs(so - do) < n:
            continue  # same-op overlap is rejected by design
        prog.append((s, d, so, do, n))
    want = [b.copy() for b in init]
    apply_reference(want, prog)
    got = [b.copy() for b in init]
    e = engine()
    with e.scope():
        for (s, d, so, do, n) in prog:
            e.segments(got[s], got[d], [so], [do], [n])
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)


def test_disable_runs_passthrough():
    src = np.arange(64, dtype=np.float32)
    dst = np.zeros(64, np.float32)
    e = engine(disable=True)
    with e.scope():
        for i in range(8):
            e.segments(src, dst, [i * 8], [i * 8], [8])
    st = e.stats()["last"]
    assert st["strategies"] == ["passthrough"] and st["calls"] == 8
    np.testing.assert_array_equal(dst, src)


def test_tuner_locks_and_profile_round_trip():
    src = np.arange(1 << 20, dtype=np.float32)
    dst = np.zeros_like(src)
    e = ce.Engine(tune_trials=2, ab_pairs=4, ab_max_pairs=8)
    for _ in range(40):  # explore runs_parallel, then A/B it against the dominance-safe merge
        with e.scope():
            for i in range(16):
                e.segments(src, dst, [i * 65536], [i * 65536], [65536])
    prof = json.loads(e.dump_profile())
    assert prof["hardware"] == ce.hardware_fingerprint()
    locked = [b["locked"] for b in prof["buckets"].values()]
    assert locked and all(locked)
    e2 = ce.Engine()
    e2.load_profile(e.dump_profile())
    locks = lambda p: {k: (v["locked"], v["alt"], v["samples"]) for k, v in p["buckets"].items()}
    assert locks(json.loads(e2.dump_profile())) == locks(prof)
    prof["hardware"] = "elsewhere"
    with pytest.raises(ValueError):
        e2.load_profile(json.dumps(prof))
    np.testing.assert_array_equal(dst, src)


def test_dependent_copies_split_into_waves():
    a = np.arange(16, dtype=np.float32)
    b = np.zeros(16, np.float32)
    c = np.zeros(16, np.float32)
    e = engine()
    with e.scope():
        e.copy_(b, a)          # wave 0
        e.copy_(c, b)          # reads b: wave 1
        e.copy_(a[:8], c[8:])  # writes a, which wave 0 read: wave 2
    assert e.stats()["last"]["waves"] == 3
    np.testing.assert_array_equal(c, np.arange(16, dtype=np.float32))
    np.testing.assert_array_equal(a[:8], np.arange(8, 16, dtype=np.float32))


def test_random_programs_match_sequential_order():
    # Overlapping copies among three buffers: the queued, merged result must equal program order.
    rng = np.random.default_rng(3)
    for trial in range(200):
        bufs = [np.arange(64, dtype=np.float32) + 100 * i for i in range(3)]
        ref = [b.copy() for b in bufs]
        prog = []
        for _ in range(int(rng.integers(1, 12))):
            s, d = rng.integers(0, 3, 2)
            n = int(rng.integers(1, 9))
            so, do = rng.integers(0, 64 - 2 * n, 2)
            step = int(rng.integers(1, 3))
            if s == d and not (so + 2 * n <= do or do + 2 * n <= so):
                continue  # one copy must not overlap itself
            prog.append((int(s), int(d), int(so), int(do), n, step))
        e = engine()
        with e.scope():
            for s, d, so, do, n, step in prog:
                e.copy_(bufs[d][do:do + n * step:step], bufs[s][so:so + n * step:step])
        for s, d, so, do, n, step in prog:
            ref[d][do:do + n * step:step] = ref[s][so:so + n * step:step].copy()
        for got, want in zip(bufs, ref):
            np.testing.assert_array_equal(got, want, err_msg=f"trial {trial}: {prog}")


def test_many_scatters_into_one_buffer_share_a_wave():
    # Every op's write bounding box spans the whole buffer, but the rows are disjoint.
    n, rows = 2048, 4
    dst = np.zeros((n * rows, 16), np.float32)
    src = np.arange(n * 16, dtype=np.float32).reshape(n, 1, 16)
    perm = np.random.default_rng(0).permutation(n * rows).reshape(n, rows)
    e = engine()
    with e.scope():
        for k in range(n):
            e.scatter_(dst, perm[k], np.broadcast_to(src[k], (rows, 16)))
    assert e.stats()["last"]["waves"] == 1
    for k in (0, n // 2, n - 1):
        np.testing.assert_array_equal(dst[perm[k]], np.broadcast_to(src[k], (rows, 16)))


def test_lazy_views_compose():
    cache = np.arange(10 * 4 * 3, dtype=np.float32).reshape(10, 4, 3)
    ids = [7, 1, 1, 4]
    src = ce.lazy(cache).index_select(0, ids).permute(0, 2, 1).flip(2)
    out = np.empty(src.shape, np.float16)
    engine().copy_(out, src)
    np.testing.assert_array_equal(out, cache[ids].transpose(0, 2, 1)[:, :, ::-1].astype(np.float16))
    dst = np.zeros((10, 3, 4), np.float32)
    engine().copy_(ce.lazy(dst).index_select(0, [2, 5]).transpose(1, 2), cache[:2])
    np.testing.assert_array_equal(dst[[2, 5]].transpose(0, 2, 1), cache[:2])


# ---- native baseline -------------------------------------------------------------------


def test_slow_native_baseline_loses_to_ce():
    src = np.arange(1 << 16, dtype=np.float32)
    dst = np.zeros_like(src)
    e = ce.Engine(tune_trials=1, ab_pairs=8, ab_max_pairs=16)

    def slow(d, s):
        time.sleep(0.002)
        np.copyto(d, s)

    e._native = slow
    for _ in range(60):
        e.copy_(dst, src)
    assert e.stats()["last"]["strategies"] != ["native"]
    assert e.stats()["native_ops"] == 0
    np.testing.assert_array_equal(dst, src)


def test_fast_native_baseline_wins_and_runs_at_interception():
    src = np.arange(1 << 22, dtype=np.float32)
    dst = np.zeros_like(src)
    calls = []
    e = ce.Engine(tune_trials=1, ab_pairs=8, ab_max_pairs=16)
    e._native = lambda d, s: calls.append(1)  # free, so it must win
    for _ in range(60):
        e.copy_(dst, src)
    st = e.stats()
    assert st["native_ops"] > 0  # locked native: later lone copies skip the queue entirely
    n = len(calls)
    e.copy_(dst, src)
    assert len(calls) == n + 1 and e.stats()["native_ops"] == st["native_ops"] + 1


def test_disable_runs_natively_in_program_order():
    a = np.arange(64, dtype=np.float32)
    b = np.zeros(64, np.float32)
    c = np.zeros(64, np.float32)
    e = ce.Engine(disable=True)
    with e.scope():
        e.segments(a, b, [0], [0], [64])  # no native form: queued
        e.copy_(c, b)                     # native now: must see the queued write to b first
        e.copy_(a, c[::-1].copy())        # native, overwrites what the queued copy reads
    np.testing.assert_array_equal(b, np.arange(64, dtype=np.float32))
    np.testing.assert_array_equal(c, np.arange(64, dtype=np.float32))
    assert e.stats()["native_ops"] == 2


def test_explore_off_takes_baseline_and_records_uncovered_buckets():
    src = np.arange(4096, dtype=np.float32).reshape(64, 64)
    dst = np.zeros((64, 64), np.float32)
    e = ce.Engine(explore=False)
    e.copy_(dst, src.T)
    np.testing.assert_array_equal(dst, src.T)
    assert e.stats()["native_ops"] == 0 and e.stats()["last"]["strategies"] == ["native"]
    prof = json.loads(e.dump_profile())
    assert prof["uncovered"] and not prof["buckets"]
    e2 = ce.Engine()
    e2.load_profile(e.dump_profile())
    assert json.loads(e2.dump_profile())["uncovered"] == prof["uncovered"]


def test_fence_flushes_only_conflicting_queued_copies():
    src = np.arange(256, dtype=np.float32)
    dst = np.zeros(256, np.float32)
    other = np.zeros(16, np.float32)
    e = engine()

    def rng(x):
        p = x.__array_interface__["data"][0]
        return ("host", 0, p, p + x.nbytes)

    with e.scope():
        e.copy_(dst, src)
        e._e.fence([rng(other)], [rng(other)])
        assert e._e.queued == 1
        e._e.fence([rng(dst)], [])  # reads what the queue writes
        assert e._e.queued == 0
        np.testing.assert_array_equal(dst, src)
        e.copy_(dst, src)
        e._e.fence([], [rng(src)])  # overwrites what the queue reads
        assert e._e.queued == 0


# ---- online sampling after a decision is locked ------------------------------------------


def _online_engine():
    return ce.Engine(tune_trials=1, ab_pairs=8, ab_max_pairs=16, online_every=2)


def test_online_sampling_leaves_native_when_it_gets_slow():
    src = np.arange(1 << 16, dtype=np.float32)
    dst = np.zeros_like(src)
    e = _online_engine()
    e._native = lambda d, s: None  # free: native locks
    for _ in range(60):
        e.copy_(dst, src)
    assert e.stats()["native_ops"] > 0

    def slow(d, s):
        time.sleep(0.002)
        np.copyto(d, s)

    e._native = slow
    for _ in range(400):
        e.copy_(dst, src)
    before = e.stats()["native_ops"]
    for _ in range(20):
        e.copy_(dst, src)
    assert e.stats()["native_ops"] == before  # ce again: nothing runs natively at interception
    on = [b["online"] for b in json.loads(e.dump_profile())["buckets"].values()]
    assert sum(o["flips"] + o["reopens"] for o in on) >= 1
    np.testing.assert_array_equal(dst, src)


def _copy_seconds(dst, src, n=20):
    np.copyto(dst, src)
    t0 = time.perf_counter()
    for _ in range(n):
        np.copyto(dst, src)
    return (time.perf_counter() - t0) / n


def test_online_sampling_moves_to_native_when_it_gets_fast():
    # Just under 1 MB: still one memcpy on the host (no worker threads), and long
    # enough that Python call overhead does not dominate the native side.
    src = np.arange(250_000, dtype=np.float32)
    dst = np.zeros_like(src)
    e = _online_engine()
    t = _copy_seconds(dst, src)

    def slower(d, s):  # ~1.6x ce's copy: loses clearly (noisy runners), but within 2x, so it keeps being re-checked
        np.copyto(d, s)
        deadline = time.perf_counter() + 0.6 * t  # sleep() is too coarse at this scale
        while time.perf_counter() < deadline:
            pass

    e._native = slower
    for _ in range(60):
        e.copy_(dst, src)
    assert e.stats()["native_ops"] == 0
    e._native = lambda d, s: None  # now free
    for _ in range(400):
        e.copy_(dst, src)
    assert e.stats()["native_ops"] > 0, e.dump_profile()
    on = [b["online"] for b in json.loads(e.dump_profile())["buckets"].values()]
    assert sum(o["flips"] + o["reopens"] for o in on) >= 1  # flipped, or the locked side drifted and re-opened


def test_online_sampling_never_runs_a_far_slower_finalist():
    # Re-checking a side many times slower would stall the run it samples.
    src = np.arange(1 << 16, dtype=np.float32)
    dst = np.zeros_like(src)
    e = _online_engine()
    calls = []

    def far(d, s):
        calls.append(1)
        time.sleep(0.01)
        np.copyto(d, s)

    e._native = far
    for _ in range(60):
        e.copy_(dst, src)
    assert e.stats()["native_ops"] == 0
    n = len(calls)
    for _ in range(200):
        e.copy_(dst, src)
    online = [b["online"] for b in json.loads(e.dump_profile())["buckets"].values()]
    assert sum(o["samples"] for o in online) > 0
    # Online re-checks never ran it. Only re-exploration may (a slow runner can
    # make the locked side drift, which re-opens the bucket by design).
    assert len(calls) == n or sum(o["reopens"] for o in online) > 0


def test_ab_stops_early_when_one_side_is_far_faster():
    src = np.arange(1 << 16, dtype=np.float32)
    dst = np.zeros_like(src)
    e = ce.Engine(tune_trials=1, ab_pairs=40, ab_max_pairs=80)
    calls = []

    def far(d, s):
        calls.append(1)
        time.sleep(0.01)
        np.copyto(d, s)

    e._native = far
    for _ in range(40):
        e.copy_(dst, src)
    assert len(calls) <= 8  # locked after ~6 pairs, not 40


def test_explicit_copies_without_native_never_run_a_huge_passthrough():
    # A gather of many rows has no torch / numpy equivalent: passthrough would be
    # one call per row, so it is neither the reference nor a candidate.
    cache = np.arange(8192 * 3 * 16, dtype=np.float32).reshape(8192, 3, 16)
    ids = np.random.default_rng(0).permutation(8192)
    out = np.zeros((8192, 16, 3), np.float32)
    e = ce.Engine(tune_trials=1, ab_pairs=4, ab_max_pairs=8, online_every=2)
    seen = set()
    for _ in range(40):
        e.copy_(out, ce.lazy(cache).index_select(0, ids).permute(0, 2, 1))
        seen.update(e.stats()["last"]["strategies"])
    assert "passthrough" not in seen
    np.testing.assert_array_equal(out, cache[ids].transpose(0, 2, 1))


def test_freeze_replays_the_decision_not_the_last_sample():
    src = np.arange(1 << 16, dtype=np.float32)
    dst = np.zeros_like(src)
    calls = []

    def native(d, s):
        calls.append(1)
        np.copyto(d, s)

    # During the A/B: the last flush may have run the challenger; freeze takes the reference (native).
    e = ce.Engine(tune_trials=1, ab_pairs=40, ab_max_pairs=80)
    e._native = native
    for _ in range(4):
        e.copy_(dst, src)
        if e.stats()["last"]["strategies"] != ["native"]:
            break
    assert e.stats()["last"]["strategies"] != ["native"]
    n = len(calls)
    e.freeze().run()
    assert len(calls) == n + 1
    # Locked to native, but the last flush was an online sample of ce: still native.
    e = ce.Engine(tune_trials=1, ab_pairs=4, ab_max_pairs=8, online_every=1)
    e._native = lambda d, s: calls.append(1)  # free: native locks
    for _ in range(200):
        e.copy_(dst, src)
        if e.stats()["native_ops"] and e.stats()["last"]["strategies"] not in (["native"], []):
            break
    n = len(calls)
    e.freeze().run()
    assert len(calls) == n + 1


def test_online_sampling_off_never_times_locked_runs():
    src = np.arange(1 << 12, dtype=np.float32)
    dst = np.zeros_like(src)
    e = ce.Engine(tune_trials=1, ab_pairs=4, ab_max_pairs=8, online_every=0)
    for _ in range(200):
        e.copy_(dst, src)
    assert all(b["online"]["samples"] == 0 for b in json.loads(e.dump_profile())["buckets"].values())


def test_profiles_from_another_version_are_not_trusted():
    src = np.arange(1 << 12, dtype=np.float32)
    dst = np.zeros_like(src)
    e = ce.Engine(tune_trials=1, ab_pairs=4, ab_max_pairs=8)
    for _ in range(40):
        e.copy_(dst, src)
    prof = json.loads(e.dump_profile())
    assert prof["version"] == ce.__version__ and prof["buckets"]
    prof["version"] = "0.0.0"
    e2 = ce.Engine()
    e2.load_profile(json.dumps(prof))
    p2 = json.loads(e2.dump_profile())
    assert not p2["buckets"] and set(p2["uncovered"]) >= set(prof["buckets"])


# ---- allocation boundaries -----------------------------------------------------------------


def _endpoint(arr, base_off, nbytes):
    p = arr.__array_interface__["data"][0]
    return ce._core.Endpoint("host", 0, p + base_off, nbytes, "float32", False)


def test_copies_never_merge_across_allocations():
    # Two allocations that happen to be adjacent in memory: one copy call may
    # not span both (the CUDA driver rejects it), so runs merge only inside one.
    buf = np.arange(256, dtype=np.float32)
    dst = np.zeros(256, np.float32)
    a, b = _endpoint(buf, 0, 512), _endpoint(buf, 512, 512)
    d = _endpoint(dst, 0, 1024)
    e = engine()
    with e.scope():
        e._e.copy_runs(a, d, [0], [0], [128], ce.Elementwise())
        e._e.copy_runs(b, d, [0], [128], [128], ce.Elementwise())
    assert e.stats()["last"]["calls"] == 2
    np.testing.assert_array_equal(dst, buf)
    # Rows alternating between the two allocations: no 2-D block across them either.
    dst[:] = 0
    e = engine()
    with e.scope():
        for r in range(8):
            src = a if r % 2 == 0 else b
            e._e.copy_runs(src, d, [(r // 2) * 16], [r * 32], [16], ce.Elementwise())
    assert e.stats()["last"]["calls"] >= 2
    want = np.zeros(256, np.float32)
    for r in range(8):
        off = (r // 2) * 16 + (0 if r % 2 == 0 else 128)
        want[r * 32:r * 32 + 16] = buf[off:off + 16]
    np.testing.assert_array_equal(dst, want)


def test_the_same_memory_described_twice_still_merges():
    buf = np.arange(256, dtype=np.float32)
    dst = np.zeros(256, np.float32)
    whole, first = _endpoint(buf, 0, 1024), _endpoint(buf, 0, 512)  # overlapping descriptions
    d = _endpoint(dst, 0, 1024)
    e = engine()
    with e.scope():
        e._e.copy_runs(first, d, [0], [0], [128], ce.Elementwise())
        e._e.copy_runs(whole, d, [128], [128], [128], ce.Elementwise())
    assert e.stats()["last"]["calls"] == 1  # the cut at 512 bytes is inside `whole`
    np.testing.assert_array_equal(dst, buf)


# ---- threads ---------------------------------------------------------------------------------


def test_default_engine_is_per_thread():
    import threading
    seen = []
    t = threading.Thread(target=lambda: seen.append(ce.default_engine()))
    t.start()
    t.join()
    assert seen[0] is not ce.default_engine()


def test_a_scope_on_another_thread_is_an_error_not_a_silent_queue():
    import threading
    e = engine()
    a = np.arange(16, dtype=np.float32)
    b = np.zeros(16, np.float32)
    entered, release = threading.Event(), threading.Event()

    def owner():
        with e.scope():
            entered.set()
            release.wait(5)

    t = threading.Thread(target=owner)
    t.start()
    entered.wait(5)
    try:
        with pytest.raises(ValueError, match="another thread"):
            e.copy_(b, a)
        with pytest.raises(ValueError, match="another thread"):
            with e.scope():
                pass
    finally:
        release.set()
        t.join()
    e.copy_(b, a)  # fine once the scope is closed
    np.testing.assert_array_equal(b, a)


def test_threads_with_their_own_engines_run_concurrently():
    import threading
    errors = []

    def work(seed):
        try:
            rng = np.random.default_rng(seed)
            for _ in range(30):
                src = rng.standard_normal((64, 32)).astype(np.float32)
                dst = np.zeros((32, 64), np.float32)
                with ce.scope():
                    for r in range(0, 64, 8):
                        ce.copy_(dst[:, r:r + 8], src[r:r + 8].T)
                np.testing.assert_array_equal(dst, src.T)
        except Exception as ex:  # noqa: BLE001 - reported below
            errors.append(ex)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, errors


def test_last_flush_explains_its_decisions():
    src = np.arange(1 << 16, dtype=np.float32)
    dst = np.zeros_like(src)
    e = ce.Engine(tune_trials=1, ab_pairs=4, ab_max_pairs=8)
    e.copy_(dst, src)
    d = e.stats()["last"]["decisions"]
    assert len(d) == 1 and d[0]["strategy"] and d[0]["why"].startswith(("exploring", "A/B")), d
    for _ in range(40):
        e.copy_(dst, src)
    e._e.flush()
    e.copy_(dst, src[::-1].copy())  # a full flush (the lone-op memo does not refresh stats)
    d = e.stats()["last"]["decisions"][0]
    assert d["why"].startswith(("locked", "online re-check")) and d["ops"] == 1 and d["bytes"] == src.nbytes, d
    assert ce.Engine(prior_only=True).copy_(dst, src) is dst


def test_warmup_runs_until_every_pattern_is_locked():
    src = np.arange(1 << 16, dtype=np.float32)
    dst = np.zeros_like(src)
    a = np.arange(4096, dtype=np.float32).reshape(64, 64)
    b = np.zeros((64, 64), np.float32)
    e = ce.Engine()

    def step(eng):
        eng.copy_(dst, src)
        eng.copy_(b, a.T)

    n = e.warmup(step)
    assert n > 4
    assert all(v["locked"] for v in json.loads(e.dump_profile())["buckets"].values())
    assert e.warmup(step, iters=3) == 3


def test_plan_cache_evicts_least_recently_used():
    # A working set one larger than the cache keeps its hot entry (a full clear lost it).
    e = ce.Engine()
    e._opts.plan_cache = 4  # not a Python option: rebuild the core engine with it
    e._e = ce._core.Engine(e._opts)
    e._handle = e._e.handle
    src, dst = np.arange(64, dtype=np.float32), np.zeros(64, np.float32)
    hot = lambda: e.segments(src, dst, [0], [0], [8])
    hot()
    for k in range(1, 12):
        e.segments(src, dst, [k], [k], [1])  # a new plan each
        hot()
        assert e.stats()["last"]["plan_cache"] == "hit", k
    assert e.stats()["plans"] <= 4
    assert e.stats()["plan_us"] > 0


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_copy_many_plans_the_pairs_together(dev):
    torch = pytest.importorskip("torch")
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    layers = 12
    host = [torch.randn(16, 64) for _ in range(layers)]
    if dev == "cuda":
        host = [h.pin_memory() for h in host]
    dst = torch.zeros(layers, 16, 64, device=dev)
    e = ce.Engine()
    e.copy_many(list(dst), host)
    assert e.stats()["last"]["ops"] == layers  # one flush for all pairs
    if dev == "cuda":
        torch.cuda.synchronize()
    assert torch.equal(dst.cpu(), torch.stack(host))
    with pytest.raises(ValueError):
        e.copy_many(list(dst), host[:-1])
    arrs = [np.zeros(8, np.float32) for _ in range(3)]
    e.copy_many(arrs, [np.full(8, float(i), np.float32) for i in range(3)])  # not tensors: pair by pair
    assert [a[0] for a in arrs] == [0.0, 1.0, 2.0]


def test_copy_many_with_dtypes_the_op_does_not_take():
    torch = pytest.importorskip("torch")
    e8 = torch.float8_e8m0fnu  # not on the stable ABI: pair by pair, the rest in one call
    srcs = [torch.randn(16), torch.randn(8).to(e8), torch.randn(4)]
    dsts = [torch.zeros(16), torch.zeros(8, dtype=e8), torch.zeros(4)]
    e = ce.Engine()
    e.copy_many(dsts, srcs)
    assert e.stats()["last"]["ops"] == 3
    for d, s in zip(dsts, srcs):
        assert torch.equal(d.view(torch.uint8) if d.dtype == e8 else d, s.view(torch.uint8) if s.dtype == e8 else s)
