"""copyengine vs torch on copy workloads taken from vLLM and SGLang serving code.

    python bench/bench_serving.py [--only substr] [--json out.json] [--reps N]

Each case reproduces one copy path of an inference engine with the shapes of a
Llama-3-8B-class model (32 layers, 8 KV heads, head_dim 128, bf16) unless noted,
scaled to fit a small GPU. Columns:

  fw         the torch code the framework runs on that path (or its torch fallback)
  alt        another torch formulation (per-copy loop, or vectorized)
  ce_scope   the fw/alt loop unchanged inside ``Engine.scope()`` (copy_ intercepted)
  ce_api     the explicit API (segments / gather / scatter_) in one scope
  ce_frozen  replay of the ce_api plan (where the case is a single flush)

Where the framework normally uses a custom kernel or the CUDA batch-memcpy API,
the case says so; fw is then the torch path, not the fastest path it has.
Prints median microseconds per iteration. Results are machine specific; keep
them out of the repository.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time

import numpy as np
import torch

import copyengine as ce

DEV = "cuda"
BF16 = torch.bfloat16
ARENA: list | None = None   # --arena: carve pinned host tensors out of one allocation


def pin(t: torch.Tensor) -> torch.Tensor:
    """Pinned copy of `t`: its own pinned allocation (what the frameworks do), or a slice of one arena."""
    if ARENA is None:
        return t.pin_memory()
    nbytes = t.numel() * t.element_size()
    buf, used = ARENA
    if used + nbytes > buf.numel():
        raise MemoryError("pinned arena exhausted; raise --arena")
    ARENA[1] = (used + nbytes + 4095) // 4096 * 4096
    out = buf[used:used + nbytes].view(t.dtype).view(t.shape)
    out.copy_(t)
    return out


def timeit(fn, reps: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


def tuned(body, rounds: int) -> ce.Engine:
    """An engine whose buckets for `body` are explored, A/B tested against the baseline and locked."""
    e = ce.Engine(tune_trials=3, tune_topk=4)
    for _ in range(rounds):
        body(e)
    torch.cuda.synchronize()
    return e


def run_case(name, desc, *, reset, check, torch_fns: dict, ce_fns: dict, frozen: str | None,
             reps: int, rounds: int) -> dict:
    row = {"case": name, "desc": desc, "us": {}, "strategies": {}, "calls": {}}
    for k, fn in torch_fns.items():
        row["us"][k] = timeit(fn, reps)
    reset()
    for k, fn in torch_fns.items():   # every torch variant must agree with the reference
        fn()
        torch.cuda.synchronize()
        check()
        reset()
    for k, body in ce_fns.items():
        e = tuned(body, rounds)
        reset()
        body(e)
        torch.cuda.synchronize()
        check()
        row["us"][k] = timeit(lambda: body(e), reps)
        last = e.stats()["last"]
        row["strategies"][k] = sorted(set(last.get("strategies", [])))
        row["calls"][k] = last.get("calls")
        if k == frozen:
            # freeze() replays the last flush, which may be an online A/B sample of another
            # strategy: freeze one that ran the engine's usual choice.
            seen = []
            for _ in range(5):
                body(e)
                seen.append(tuple(e.stats()["last"].get("strategies", [])))
            usual = max(set(seen), key=seen.count)
            for _ in range(20):
                reset()
                body(e)
                if tuple(e.stats()["last"].get("strategies", [])) == usual:
                    break
            plan = e.freeze()
            row["us"]["ce_frozen"] = timeit(plan.run, reps)
            reset()
            plan.run()
            torch.cuda.synchronize()
            check()
    return row


def exact(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def rng(seed=0):
    return np.random.default_rng(seed)


def runs_of(ids: np.ndarray):
    """Split sorted-by-position (src, dst) index pairs into contiguous runs, like sglang's transfer_kv_direct."""
    s, d = ids
    brk = np.flatnonzero((np.diff(s) != 1) | (np.diff(d) != 1)) + 1
    starts = np.concatenate([[0], brk])
    lens = np.diff(np.concatenate([starts, [len(s)]]))
    return list(zip(s[starts].tolist(), d[starts].tolist(), lens.tolist()))


def paged_ids(n_tokens: int, page: int, n_slots_a: int, n_slots_b: int, r):
    """Token indices allocated in pages of `page` tokens, scattered over two pools (a, b)."""
    n_pages = n_tokens // page
    pa = r.choice(n_slots_a // page, n_pages, replace=False)
    pb = r.choice(n_slots_b // page, n_pages, replace=False)
    off = np.arange(page)
    a = (pa[:, None] * page + off).reshape(-1)
    b = (pb[:, None] * page + off).reshape(-1)
    return a.astype(np.int64), b.astype(np.int64)


# ---- vLLM -------------------------------------------------------------------------------


def case_vllm_offload(reps, rounds, direction="load", L=32, H=8, D=128, T=16, NB=512, NC=256, n=64):
    """vLLM OffloadingConnector / SimpleCPUOffload: (layer x block) pages between per-layer GPU
    caches [NB, 2*T*H*D] and pinned per-layer host chunks. vLLM issues these as one
    cuMemcpyBatchAsync (one descriptor per layer x block, CUDA >= 12.8; else a cudaMemcpyAsync loop),
    or a UVA Triton kernel for small H2D pages. fw = that loop of per-page copies; alt = the NIXL
    host-buffer torch path (vllm/distributed/kv_transfer/kv_connector/utils.py copy_kv_blocks)."""
    page = 2 * T * H * D
    gpu = [torch.randn(NB, page, dtype=BF16, device=DEV) for _ in range(L)]
    host = [pin(torch.randn(NC, page, dtype=BF16)) for _ in range(L)]
    r = rng(1)
    gid = r.choice(NB, n, replace=False)
    cid = r.choice(NC, n, replace=False)
    gl, cl = gid.tolist(), cid.tolist()
    gid_t, cid_t = torch.from_numpy(gid), torch.from_numpy(cid)
    gid_d = gid_t.to(DEV)
    ref_g = [g[gid_d].clone() for g in gpu]
    ref_h = [h[cid_t].clone() for h in host]
    load = direction == "load"

    def reset():
        for l in range(L):
            if load:
                gpu[l][gid_d] = 0
            else:
                host[l][cid_t] = 0

    def check():
        for l in range(L):
            if load:
                exact(gpu[l][gid_d], ref_h[l].to(DEV))
            else:
                exact(host[l][cid_t], ref_g[l].cpu())

    def loop():
        for l in range(L):
            g, h = gpu[l], host[l]
            for i in range(n):
                if load:
                    g[gl[i]].copy_(h[cl[i]], non_blocking=True)
                else:
                    h[cl[i]].copy_(g[gl[i]], non_blocking=True)

    def nixl():
        for l in range(L):
            if load:
                gpu[l][gid_d] = host[l][cid_t].to(DEV)
            else:
                host[l][cid_t] = gpu[l][gid_d].cpu()

    def ce_scope(e):
        with e.scope():
            loop()

    so = (cid if load else gid) * page
    do = (gid if load else cid) * page
    ln = [page] * n

    def ce_api(e):
        with e.scope():
            for l in range(L):
                if load:
                    e.segments(host[l], gpu[l], so, do, ln)
                else:
                    e.segments(gpu[l], host[l], so, do, ln)

    kb = page * 2 // 1024
    return run_case(f"vllm_offload_{direction}_{kb}k", f"{L}x{n} pages of {kb} KB",
                    reset=reset, check=check, torch_fns={"fw": loop, "alt": nixl},
                    ce_fns={"ce_scope": ce_scope, "ce_api": ce_api}, frozen="ce_api", reps=reps, rounds=rounds)


def case_vllm_lora(reps, rounds, L=32, max_loras=4, max_rank=64, rank=16, hidden=4096, inter=14336,
                   kv=1024):
    """vLLM set_lora: per (layer, module) two strided H2D copies of a pinned adapter into slot
    `index` of stacked buffers A [max_loras, 1, max_rank, in] and B [max_loras, 1, out, max_rank]."""
    mods = {"q": (hidden, hidden), "k": (hidden, kv), "v": (hidden, kv), "o": (hidden, hidden),
            "gate": (hidden, inter), "up": (hidden, inter), "down": (inter, hidden)}
    A = {(l, m): torch.zeros(max_loras, 1, max_rank, i, dtype=BF16, device=DEV) for l in range(L)
         for m, (i, o) in mods.items()}
    B = {(l, m): torch.zeros(max_loras, 1, o, max_rank, dtype=BF16, device=DEV) for l in range(L)
         for m, (i, o) in mods.items()}
    srcA = {k: pin(torch.randn(rank, v.shape[3], dtype=BF16)) for k, v in A.items()}
    srcB = {k: pin(torch.randn(v.shape[2], rank, dtype=BF16)) for k, v in B.items()}
    slot = 1

    def reset():
        for k in A:
            A[k][slot].zero_()
            B[k][slot].zero_()

    def check():
        for k in A:
            exact(A[k][slot, 0, :rank], srcA[k].to(DEV))
            exact(B[k][slot, 0, :, :rank], srcB[k].to(DEV))

    def fw():
        for k in A:
            a, b = srcA[k], srcB[k]
            A[k][slot, 0, :a.shape[0], :a.shape[1]].copy_(a, non_blocking=True)
            B[k][slot, 0, :b.shape[0], :b.shape[1]].copy_(b, non_blocking=True)

    def ce_scope(e):
        with e.scope():
            fw()

    return run_case("vllm_lora_load", f"{L}x{len(mods)}x2 strided H2D, rank {rank}/{max_rank}",
                    reset=reset, check=check, torch_fns={"fw": fw}, ce_fns={"ce_scope": ce_scope},
                    frozen="ce_scope", reps=reps, rounds=rounds)


def case_step_metadata(reps, rounds, R=256, max_reqs=1024, max_blocks=512, new_per_req=1):
    """Per-step input H2D (vLLM gpu_input_batch / block_table, SGLang ScheduleBatch): ~8 pinned
    metadata arrays plus a ragged per-request block-table append. vLLM v2 runner applies the
    block-table diff with one UVA Triton kernel; fw = the per-request torch fallback;
    alt = vectorized torch (H2D the diff, then index_put_)."""
    r = rng(2)
    meta_h = [pin(torch.from_numpy(r.integers(0, 1 << 20, R + 1 if i % 3 == 0 else R)).to(
        torch.int32 if i % 2 else torch.int64)) for i in range(8)]
    meta_d = [torch.empty_like(t, device=DEV) for t in meta_h]
    bt = torch.zeros(max_reqs, max_blocks, dtype=torch.int32, device=DEV)
    rows = r.choice(max_reqs, R, replace=False)
    cols = r.integers(0, max_blocks - new_per_req, R)
    vals = pin(torch.from_numpy(r.integers(1, 1 << 20, (R, new_per_req)).astype(np.int32)))
    rl, cl = rows.tolist(), cols.tolist()
    flat = pin(torch.from_numpy((rows[:, None] * max_blocks + cols[:, None] + np.arange(new_per_req)).reshape(-1)))

    def reset():
        bt.zero_()
        for t in meta_d:
            t.zero_()

    ref = torch.zeros_like(bt).cpu()
    for i in range(R):
        ref[rl[i], cl[i]:cl[i] + new_per_req] = vals[i]

    def check():
        exact(bt.cpu(), ref)
        for a, b in zip(meta_d, meta_h):
            exact(a.cpu(), b)

    def fw():
        for a, b in zip(meta_d, meta_h):
            a.copy_(b, non_blocking=True)
        for i in range(R):
            bt[rl[i], cl[i]:cl[i] + new_per_req].copy_(vals[i], non_blocking=True)

    def vec():
        for a, b in zip(meta_d, meta_h):
            a.copy_(b, non_blocking=True)
        bt.view(-1).index_put_((flat.to(DEV, non_blocking=True),), vals.view(-1).to(DEV, non_blocking=True))

    def ce_scope(e):
        with e.scope():
            fw()

    return run_case("step_metadata_h2d", f"8 arrays + {R} block-table appends",
                    reset=reset, check=check, torch_fns={"fw": fw, "alt": vec}, ce_fns={"ce_scope": ce_scope},
                    frozen="ce_scope", reps=reps, rounds=rounds)


# ---- SGLang -----------------------------------------------------------------------------


def case_hicache_load(reps, rounds, layout="layer_first", L=32, H=8, D=128, S=16384, HS=4096, n=2048,
                      page=16):
    """SGLang HiCache L2 -> L1 load_to_device_per_layer: token rows (H*D bf16 = 2 KB) from the
    pinned host pool into per-layer k_buffer/v_buffer [S, H, D], one layer at a time (overlapped
    with forward). The default io backend is an sgl_kernel UVA gather kernel; fw = the "direct"
    backend (transfer_kv_direct: merge contiguous runs, copy_ per run per layer) for layer_first,
    and the torch index path for page_first [2, HS, L, H, D] (strided host rows)."""
    kb = [torch.zeros(S, H, D, dtype=BF16, device=DEV) for _ in range(L)]
    vb = [torch.zeros(S, H, D, dtype=BF16, device=DEV) for _ in range(L)]
    if layout == "layer_first":
        host = pin(torch.randn(2, L, HS, H, D, dtype=BF16))
        hk = lambda kv, l: host[kv, l]          # noqa: E731  [HS, H, D] contiguous
    else:
        host = pin(torch.randn(2, HS, L, H, D, dtype=BF16))
        hk = lambda kv, l: host[kv, :, l]       # noqa: E731  [HS, H, D] strided
    hid, did = paged_ids(n, page, HS, S, rng(3))
    order = np.argsort(hid, kind="stable")    # sglang sorts host indices to lengthen runs
    hid, did = hid[order], did[order]
    runs = runs_of((hid, did))
    hid_t, did_d = torch.from_numpy(hid), torch.from_numpy(did).to(DEV)

    def reset():
        for l in range(L):
            kb[l][did_d] = 0
            vb[l][did_d] = 0

    def check():
        for l in range(L):
            exact(kb[l][did_d].cpu(), hk(0, l)[hid_t])
            exact(vb[l][did_d].cpu(), hk(1, l)[hid_t])

    def direct_layer(l):
        for s, d, m in runs:
            kb[l][d:d + m].copy_(hk(0, l)[s:s + m], non_blocking=True)
            vb[l][d:d + m].copy_(hk(1, l)[s:s + m], non_blocking=True)

    def index_layer(l):
        kb[l][did_d] = hk(0, l)[hid_t].to(DEV, non_blocking=True)
        vb[l][did_d] = hk(1, l)[hid_t].to(DEV, non_blocking=True)

    def fw():
        for l in range(L):
            (direct_layer if layout == "layer_first" else index_layer)(l)

    def alt():
        for l in range(L):
            (index_layer if layout == "layer_first" else direct_layer)(l)

    def ce_scope(e):
        for l in range(L):                    # one flush per layer, like the per-layer load
            with e.scope():
                direct_layer(l)

    row_el = H * D
    def ce_api(e):
        for l in range(L):
            with e.scope():
                for kv, buf in ((0, kb), (1, vb)):
                    v = hk(kv, l)   # a strided view: address the whole pool
                    e.segments(host, buf[l], v.storage_offset() + hid * v.stride(0), did * row_el, [row_el] * n)

    return run_case(f"sglang_hicache_load_{layout}", f"{L} layers x {n} tokens x K,V, page {page}",
                    reset=reset, check=check, torch_fns={"fw": fw, "alt": alt},
                    ce_fns={"ce_scope": ce_scope, "ce_api": ce_api}, frozen=None, reps=reps, rounds=rounds)


def case_hicache_backup(reps, rounds, L=32, H=8, D=128, S=16384, HS=4096, n=2048, page=16):
    """SGLang HiCache L1 -> L2 backup_from_device_all_layer into the default page_first host
    layout [2, HS, L, H, D]: all layers at once, D2H. Default backend: sgl_kernel UVA kernel
    (staged through a GPU buffer for page_first); fw = the direct backend's run-merging copy_ loop."""
    kb = [torch.randn(S, H, D, dtype=BF16, device=DEV) for _ in range(L)]
    vb = [torch.randn(S, H, D, dtype=BF16, device=DEV) for _ in range(L)]
    host = pin(torch.zeros(2, HS, L, H, D, dtype=BF16))
    did, hid = paged_ids(n, page, S, HS, rng(4))
    order = np.argsort(did, kind="stable")
    did, hid = did[order], hid[order]
    runs = runs_of((did, hid))
    hid_t, did_d = torch.from_numpy(hid), torch.from_numpy(did).to(DEV)

    def reset():
        host[:, hid_t] = 0

    def check():
        for l in range(L):
            exact(host[0, hid_t, l], kb[l][did_d].cpu())
            exact(host[1, hid_t, l], vb[l][did_d].cpu())

    def fw():
        for l in range(L):
            for s, d, m in runs:
                host[0, d:d + m, l].copy_(kb[l][s:s + m], non_blocking=True)
                host[1, d:d + m, l].copy_(vb[l][s:s + m], non_blocking=True)

    def alt():
        for l in range(L):
            host[0, hid_t, l] = kb[l][did_d].cpu()
            host[1, hid_t, l] = vb[l][did_d].cpu()

    def ce_scope(e):
        with e.scope():
            fw()

    row_el = H * D

    def ce_api(e):
        with e.scope():
            for l in range(L):
                for kv, buf in ((0, kb), (1, vb)):
                    v = host[kv, :, l]
                    e.segments(buf[l], host, did * row_el, v.storage_offset() + hid * v.stride(0), [row_el] * n)

    return run_case("sglang_hicache_backup_pf", f"{L} layers x {n} tokens x K,V -> page_first",
                    reset=reset, check=check, torch_fns={"fw": fw, "alt": alt},
                    ce_fns={"ce_scope": ce_scope, "ce_api": ce_api}, frozen="ce_api", reps=reps, rounds=rounds)


def case_set_kv(reps, rounds, n=256, L=32, H=8, D=128, S=65536, quant=False):
    """SGLang MHATokenToKVPool.set_kv_buffer torch fallback: per layer k_cache[loc] = k (index_put
    by out_cache_loc). Frameworks use a store kernel (sglang store_cache / vLLM reshape_and_cache).
    quant=True stores int8 with a per-tensor scale (a stand-in for fp8 e4m3, which ce lacks)."""
    dt = torch.int8 if quant else BF16
    kc = [torch.zeros(S, H, D, dtype=dt, device=DEV) for _ in range(L)]
    k = [torch.randn(n, H, D, dtype=BF16, device=DEV) * 3 for _ in range(L)]
    loc = rng(5).choice(S, n, replace=False).astype(np.int64)
    loc_d = torch.from_numpy(loc).to(DEV)
    s = 0.05
    fn = ce.Elementwise(scale=1 / s, round=True, clamp=(-127, 127)) if quant else None

    def q(x):
        return torch.clamp(torch.round(x.float() * (1 / s)), -127, 127).to(torch.int8) if quant else x

    ref = [q(x) for x in k]

    def reset():
        for c in kc:
            c[loc_d] = 0

    def check():
        for c, x in zip(kc, ref):
            exact(c[loc_d], x)

    def fw():
        for c, x in zip(kc, k):
            c[loc_d] = q(x)

    def ce_api(e):
        for c, x in zip(kc, k):              # one store per layer, between that layer's kernels
            with e.scope():
                e.scatter_(c, loc, x, fn=fn)

    tag = "int8" if quant else "bf16"
    return run_case(f"set_kv_{tag}_n{n}", f"{L} layers, scatter {n} tokens by out_cache_loc",
                    reset=reset, check=check, torch_fns={"fw": fw}, ce_fns={"ce_api": ce_api},
                    frozen=None, reps=reps, rounds=rounds)


def case_move_kv(reps, rounds, n=64, L=32, H=8, D=128, S=65536):
    """SGLang move_kv_cache (spec-decode accept / compaction) torch path: per layer
    k[tgt] = k[src]; v[tgt] = v[src]. CUDA default: one Triton launch over a pointer table."""
    kb = [torch.randn(S, H, D, dtype=BF16, device=DEV) for _ in range(L)]
    vb = [torch.randn(S, H, D, dtype=BF16, device=DEV) for _ in range(L)]
    sel = rng(6).choice(S, 2 * n, replace=False).astype(np.int64)
    src, tgt = sel[:n], sel[n:]
    src_d, tgt_d = torch.from_numpy(src).to(DEV), torch.from_numpy(tgt).to(DEV)
    ref = [(k[src_d].clone(), v[src_d].clone()) for k, v in zip(kb, vb)]

    def reset():
        for k, v in zip(kb, vb):
            k[tgt_d] = 0
            v[tgt_d] = 0

    def check():
        for (k, v), (rk, rv) in zip(zip(kb, vb), ref):
            exact(k[tgt_d], rk)
            exact(v[tgt_d], rv)

    def fw():
        for k, v in zip(kb, vb):
            k[tgt_d] = k[src_d]
            v[tgt_d] = v[src_d]

    def ce_api(e):
        with e.scope():
            for k, v in zip(kb, vb):
                e.scatter_(k, tgt, ce.lazy(k).index_select(0, src))
                e.scatter_(v, tgt, ce.lazy(v).index_select(0, src))

    return run_case(f"sglang_move_kv_n{n}", f"{L} layers x K,V, {n} token slots D2D",
                    reset=reset, check=check, torch_fns={"fw": fw}, ce_fns={"ce_api": ce_api},
                    frozen="ce_api", reps=reps, rounds=rounds)


def case_pd_staging(reps, rounds, L=32, H=8, D=128, S=16384, n=4096, nh=4, h0=4):
    """SGLang PD disaggregation with heterogeneous TP (staging_buffer._gather_all_layers_torch):
    gather a head slice of scattered tokens of every layer into one contiguous staging buffer
    for a single RDMA. Default is a Triton kernel; fw = the torch path."""
    kb = [torch.randn(S, H, D, dtype=BF16, device=DEV) for _ in range(L)]
    vb = [torch.randn(S, H, D, dtype=BF16, device=DEV) for _ in range(L)]
    stg = torch.empty(2, L, n, nh, D, dtype=BF16, device=DEV)
    idx = rng(7).choice(S, n, replace=False).astype(np.int64)
    idx_d = torch.from_numpy(idx).to(DEV)
    gidx = idx_d.view(-1, 1, 1).expand(n, nh, D)

    def reset():
        stg.zero_()

    def check():
        for l in range(L):
            exact(stg[0, l], kb[l][idx_d, h0:h0 + nh])
            exact(stg[1, l], vb[l][idx_d, h0:h0 + nh])

    def fw():
        for kv, buf in ((0, kb), (1, vb)):
            for l in range(L):
                torch.gather(buf[l][:, h0:h0 + nh], 0, gidx, out=stg[kv, l])

    def ce_api(e):
        with e.scope():
            for kv, buf in ((0, kb), (1, vb)):
                for l in range(L):
                    e.copy_(stg[kv, l], ce.lazy(buf[l]).index_select(0, idx).slice(1, h0, h0 + nh))

    return run_case("sglang_pd_staging_gather", f"{L} layers x K,V, {n} tokens, heads {h0}:{h0 + nh}",
                    reset=reset, check=check, torch_fns={"fw": fw}, ce_fns={"ce_api": ce_api},
                    frozen="ce_api", reps=reps, rounds=rounds)


def case_mamba_cow(reps, rounds, L=24, slots=17, HV=32, V=128, K=128, conv_dim=8192, kc=3, n=4):
    """SGLang MambaPool.copy_from (prefix-cache copy-on-write of hybrid-model state), all layers
    at once: conv[:, dst] = conv[:, src]; temporal[:, dst] = temporal[:, src] (Qwen3-Next-like GDN)."""
    conv = torch.randn(L, slots, conv_dim, kc, dtype=BF16, device=DEV)
    temp = torch.randn(L, slots, HV, V, K, dtype=torch.float32, device=DEV)
    sel = rng(8).choice(slots, 2 * n, replace=False).astype(np.int64)
    src, dst = sel[:n], sel[n:]
    src_d, dst_d = torch.from_numpy(src).to(DEV), torch.from_numpy(dst).to(DEV)
    rc, rt = conv[:, src_d].clone(), temp[:, src_d].clone()

    def reset():
        conv[:, dst_d] = 0
        temp[:, dst_d] = 0

    def check():
        exact(conv[:, dst_d], rc)
        exact(temp[:, dst_d], rt)

    def fw():
        conv[:, dst_d] = conv[:, src_d]
        temp[:, dst_d] = temp[:, src_d]

    def ce_api(e):
        with e.scope():
            e.scatter_(conv, dst, ce.lazy(conv).index_select(1, src), dim=1)
            e.scatter_(temp, dst, ce.lazy(temp).index_select(1, src), dim=1)

    return run_case("sglang_mamba_cow", f"{L} layers, {n} slots, {HV * V * K * 4 >> 20} MB/slot/layer",
                    reset=reset, check=check, torch_fns={"fw": fw}, ce_fns={"ce_api": ce_api},
                    frozen="ce_api", reps=reps, rounds=rounds)


def case_moe_lora(reps, rounds, L=16, E=64, max_loras=2, max_rank=32, rank=16, hidden=2048, inter=768):
    """SGLang LoRA memory pool, MoE adapter (load_lora_weight_to_buffer / copy_weight_into_buffer):
    per (layer, expert, module) strided H2D copies of pinned weights into slot `buffer_id`."""
    shapes = {"gate_A": (max_rank, hidden), "up_A": (max_rank, hidden), "down_A": (max_rank, inter),
              "gate_B": (inter, max_rank), "up_B": (inter, max_rank), "down_B": (hidden, max_rank)}
    buf = {(l, m): torch.zeros(max_loras, E, *s, dtype=BF16, device=DEV) for l in range(L)
           for m, s in shapes.items()}
    src = {}
    for (l, m), b in buf.items():
        r_, c_ = b.shape[2:]
        src[(l, m)] = pin(torch.randn(E, *((rank, c_) if m.endswith("A") else (r_, rank)), dtype=BF16))
    slot = 1

    def view_of(l, m, e):
        b = buf[(l, m)][slot, e]
        return b[:rank] if m.endswith("A") else b[:, :rank]

    def reset():
        for b in buf.values():
            b[slot].zero_()

    def check():
        for (l, m), s in src.items():
            b = buf[(l, m)][slot]
            got = b[:, :rank] if m.endswith("A") else b[:, :, :rank]
            exact(got, s.to(DEV))

    def fw():
        for (l, m), s in src.items():
            for e in range(E):
                view_of(l, m, e).copy_(s[e], non_blocking=True)

    def ce_scope(e_):
        with e_.scope():
            fw()

    return run_case("sglang_moe_lora_load", f"{L}x{E}x{len(shapes)} strided H2D, rank {rank}",
                    reset=reset, check=check, torch_fns={"fw": fw}, ce_fns={"ce_scope": ce_scope},
                    frozen="ce_scope", reps=reps, rounds=rounds)


CASES = [
    ("vllm_offload_load_64k", lambda r, k: case_vllm_offload(r, k, "load")),
    ("vllm_offload_store_64k", lambda r, k: case_vllm_offload(r, k, "store")),
    ("vllm_offload_load_8k", lambda r, k: case_vllm_offload(r, k, "load", H=1, n=256)),   # TP=8 shard
    ("vllm_lora_load", case_vllm_lora),
    ("step_metadata_h2d", case_step_metadata),
    ("sglang_hicache_load_layer_first", lambda r, k: case_hicache_load(r, k, "layer_first")),
    ("sglang_hicache_load_page_first", lambda r, k: case_hicache_load(r, k, "page_first")),
    ("sglang_hicache_backup_pf", case_hicache_backup),
    ("set_kv_bf16_n256", lambda r, k: case_set_kv(r, k, 256)),
    ("set_kv_bf16_n4096", lambda r, k: case_set_kv(r, k, 4096)),
    ("set_kv_int8_n256", lambda r, k: case_set_kv(r, k, 256, quant=True)),
    ("sglang_move_kv_n64", case_move_kv),
    ("sglang_pd_staging_gather", case_pd_staging),
    ("sglang_mamba_cow", case_mamba_cow),
    ("sglang_moe_lora_load", case_moe_lora),
]

COLS = ["fw", "alt", "ce_scope", "ce_api", "ce_frozen"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", help="run cases whose name contains this (repeatable)")
    ap.add_argument("--json", help="also write results here")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=40, help="tuning rounds per ce variant")
    ap.add_argument("--arena", type=int, default=0, metavar="MB",
                    help="allocate pinned host tensors from one arena of this size instead of one allocation each")
    args = ap.parse_args()
    global ARENA
    if args.arena:
        ARENA = [torch.empty(args.arena << 20, dtype=torch.uint8).pin_memory(), 0]
    if not torch.cuda.is_available():
        print("needs a CUDA GPU", file=sys.stderr)
        return 1
    rows = []
    hdr = f"{'case':34s}" + "".join(f"{c:>11s}" for c in COLS) + "   fw/ce  strategies (calls)"
    print(hdr, flush=True)
    for name, fn in CASES:
        if args.only and not any(s in name for s in args.only):
            continue
        try:
            r = fn(args.reps, args.rounds)
        except Exception as exc:  # report and keep going: one unsupported pattern must not hide the rest
            print(f"{name:34s}  FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:120]}", flush=True)
            rows.append({"case": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        finally:
            if ARENA is not None:
                ARENA[1] = 0
            gc.collect()
            torch.cuda.empty_cache()
        us = r["us"]
        cells = "".join(f"{us[c]:11.1f}" if c in us else f"{'-':>11s}" for c in COLS)
        ce_keys = [c for c in ("ce_scope", "ce_api") if c in us]
        best = min(ce_keys, key=lambda c: us[c])
        strat = f"{','.join(r['strategies'][best])} ({r['calls'][best]})"
        print(f"{r['case']:34s}{cells}  {us['fw'] / us[best]:6.2f}x  {strat}", flush=True)
        rows.append(r)
    if args.json:
        out = {"hardware": ce.hardware_fingerprint(), "torch": torch.__version__, "copyengine": ce.__version__,
               "unit": "us", "rows": rows}
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
