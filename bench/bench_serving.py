"""copyengine vs torch on copy workloads taken from vLLM and SGLang serving code.

    python bench/bench_serving.py [--only substr] [--json out.json] [--reps N]

Each case reproduces one copy path of an inference engine with the shapes of a
Llama-3-8B-class model (32 layers, 8 KV heads, head_dim 128, bf16) unless noted,
scaled to fit a small GPU. Every call is a new serving step: its block / token
ids are another of 8 sets (made beforehand: the scheduler's work), so nothing
can be planned for one set of addresses. Columns:

  fw         the torch code the framework runs on that path (or its torch fallback)
  alt        another torch formulation (per-copy loop, or vectorized)
  ce_scope   the fw/alt loop unchanged inside ``Engine.scope()`` (copy_ intercepted)
  ce_api     the explicit API (segments / gather / scatter_ / lazy views), called every step
  ce_plan    the step as a step plan (``Engine.plan``), the ids passed as arguments
  ce_frozen  replay of a frozen ce_api flush, timed only where the replay is still right with the
             next step's ids (where it is not: "fixed ids", the number would not mean anything)

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
STEPS = 8                   # id sets a case cycles through


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


class Steps:
    """The id sets a case cycles through: each call takes the next one (a new serving step)."""

    def __init__(self, make, k: int = STEPS):
        self.sets = [make(np.random.default_rng(1000 + i)) for i in range(k)]
        self.i = 0

    @property
    def v(self):
        return self.sets[self.i]

    def next(self):
        self.i = (self.i + 1) % len(self.sets)
        return self.sets[self.i]


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


def tuned(body, steps: Steps, rounds: int) -> ce.Engine:
    """An engine whose buckets for `body` are explored, A/B tested against the baseline and locked."""
    e = ce.Engine(tune_trials=3, tune_topk=4)
    for _ in range(rounds):
        body(e, steps.next())
    torch.cuda.synchronize()
    return e


def run_case(name, desc, *, steps: Steps, reset, check, torch_fns: dict, ce_fns: dict, frozen: str | None,
             plan=None, reps: int, rounds: int) -> dict:
    """`reset(v)`, `check(v)`, torch_fns {name: fn(v)}, ce_fns {name: body(e, v)}; `plan`:
    (make(e) -> step(*args), args(v)) for the ce_plan column (the step names the case's tensors
    itself: a step plan reaches tensors through the step's own closure); `frozen`: the ce_fns
    body to freeze."""
    row = {"case": name, "desc": desc, "us": {}, "strategies": {}, "calls": {}, "notes": {}}

    def verify(run):
        """`run(v)` with the next step's ids, from a reset, checked."""
        v = steps.next()
        reset(v)
        run(v)
        torch.cuda.synchronize()
        check(v)

    for k, fn in torch_fns.items():
        row["us"][k] = timeit(lambda: fn(steps.next()), reps)
        verify(fn)   # every torch variant must agree with the reference
    for k, body in ce_fns.items():
        e = tuned(body, steps, rounds)
        verify(lambda v: body(e, v))
        row["us"][k] = timeit(lambda: body(e, steps.next()), reps)
        last = e.stats()["last"]
        row["strategies"][k] = sorted(set(last.get("strategies", [])))
        row["calls"][k] = last.get("calls")
        if k == frozen:
            try:
                body(e, steps.next())
                fr = e.freeze()
                for _ in range(3):  # a frozen flush must give each new id set's result, not one by chance
                    verify(lambda v: fr.run())
            except AssertionError:
                row["notes"]["ce_frozen"] = "fixed ids: a frozen flush replays the ids it was frozen with"
            except Exception as exc:  # reported, the other columns still count
                row["notes"]["ce_frozen"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:100]}"
            else:
                row["us"]["ce_frozen"] = timeit(lambda: (steps.next(), fr.run()), reps)
    if plan is not None:
        make_step, args = plan
        try:
            e = ce.Engine(tune_trials=3, tune_topk=4)
            p = e.plan(make_step(e))
            for _ in range(rounds):
                p(*args(steps.next()))
            torch.cuda.synchronize()
            verify(lambda v: p(*args(v)))
            row["us"]["ce_plan"] = timeit(lambda: p(*args(steps.next())), reps)
            last = e.stats()["last"]
            row["strategies"]["ce_plan"] = sorted(set(last.get("strategies", [])))
            row["calls"]["ce_plan"] = last.get("calls")
            row["notes"]["ce_plan"] = _plan_state(p)
        except Exception as exc:  # reported, the other columns still count
            row["notes"]["ce_plan"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:100]}"
    row["notes"] = {k: n for k, n in row["notes"].items() if n}
    return row


def _plan_state(p) -> str:
    """How a step plan runs the step: its traces (one per id set means it retraced each step),
    how many replay, and why the others run the step plainly."""
    traces = [t for lay in getattr(p, "_layouts", {}).values() for t in lay["traces"]]
    plain = sorted({t.plain for t in traces if t.plain})
    replay = sum(1 for t in traces if not t.plain)
    if replay == len(traces) and len(traces) <= 2:
        return ""
    return f"{len(traces)} traces, {replay} replay" + (f"; plain: {'; '.join(plain)[:120]}" if plain else "")


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
    load = direction == "load"

    def make(r):
        gid, cid = r.choice(NB, n, replace=False), r.choice(NC, n, replace=False)
        return {"gl": gid.tolist(), "cl": cid.tolist(), "gid_d": torch.from_numpy(gid).to(DEV),
                "cid_t": torch.from_numpy(cid), "so": (cid if load else gid) * page, "do": (gid if load else cid) * page}
    steps = Steps(make)
    ln = [page] * n

    def reset(v):
        for l in range(L):
            if load:
                gpu[l][v["gid_d"]] = 0
            else:
                host[l][v["cid_t"]] = 0

    def check(v):
        for l in range(L):
            if load:
                exact(gpu[l][v["gid_d"]], host[l][v["cid_t"]].to(DEV))
            else:
                exact(host[l][v["cid_t"]], gpu[l][v["gid_d"]].cpu())

    def loop(v):
        gl, cl = v["gl"], v["cl"]
        for l in range(L):
            g, h = gpu[l], host[l]
            for i in range(n):
                if load:
                    g[gl[i]].copy_(h[cl[i]], non_blocking=True)
                else:
                    h[cl[i]].copy_(g[gl[i]], non_blocking=True)

    def nixl(v):
        for l in range(L):
            if load:
                gpu[l][v["gid_d"]] = host[l][v["cid_t"]].to(DEV)
            else:
                host[l][v["cid_t"]] = gpu[l][v["gid_d"]].cpu()

    def ce_scope(e, v):
        with e.scope():
            loop(v)

    def ce_api(e, v):
        with e.scope():
            for l in range(L):
                if load:
                    e.segments(host[l], gpu[l], v["so"], v["do"], ln)
                else:
                    e.segments(gpu[l], host[l], v["so"], v["do"], ln)

    def step(e):   # GPU block ids on the GPU, host chunk ids on the host
        def run(gid_d, cid_t):
            for l in range(L):
                if load:
                    e.copy_(ce.lazy(gpu[l])[gid_d], ce.lazy(host[l])[cid_t])
                else:
                    e.copy_(ce.lazy(host[l])[cid_t], ce.lazy(gpu[l])[gid_d])
        return run

    kb = page * 2 // 1024
    return run_case(f"vllm_offload_{direction}_{kb}k", f"{L}x{n} pages of {kb} KB", steps=steps,
                    reset=reset, check=check, torch_fns={"fw": loop, "alt": nixl},
                    ce_fns={"ce_scope": ce_scope, "ce_api": ce_api}, frozen="ce_api",
                    plan=(step, lambda v: (v["gid_d"], v["cid_t"])), reps=reps, rounds=rounds)


def case_vllm_lora(reps, rounds, L=32, max_loras=4, max_rank=64, rank=16, hidden=4096, inter=14336,
                   kv=1024):
    """vLLM set_lora: per (layer, module) two strided H2D copies of a pinned adapter into slot
    `index` of stacked buffers A [max_loras, 1, max_rank, in] and B [max_loras, 1, out, max_rank].
    The slot an adapter lands in changes from load to load."""
    mods = {"q": (hidden, hidden), "k": (hidden, kv), "v": (hidden, kv), "o": (hidden, hidden),
            "gate": (hidden, inter), "up": (hidden, inter), "down": (inter, hidden)}
    A = {(l, m): torch.zeros(max_loras, 1, max_rank, i, dtype=BF16, device=DEV) for l in range(L)
         for m, (i, o) in mods.items()}
    B = {(l, m): torch.zeros(max_loras, 1, o, max_rank, dtype=BF16, device=DEV) for l in range(L)
         for m, (i, o) in mods.items()}
    srcA = {k: pin(torch.randn(rank, v.shape[3], dtype=BF16)) for k, v in A.items()}
    srcB = {k: pin(torch.randn(v.shape[2], rank, dtype=BF16)) for k, v in B.items()}
    steps = Steps(lambda r: {"slot": int(r.integers(0, max_loras))})

    def reset(v):
        for k in A:
            A[k][v["slot"]].zero_()
            B[k][v["slot"]].zero_()

    def check(v):
        for k in A:
            exact(A[k][v["slot"], 0, :rank], srcA[k].to(DEV))
            exact(B[k][v["slot"], 0, :, :rank], srcB[k].to(DEV))

    def fw(v):
        slot = v["slot"]
        for k in A:
            a, b = srcA[k], srcB[k]
            A[k][slot, 0, :a.shape[0], :a.shape[1]].copy_(a, non_blocking=True)
            B[k][slot, 0, :b.shape[0], :b.shape[1]].copy_(b, non_blocking=True)

    def ce_scope(e, v):
        with e.scope():
            fw(v)

    return run_case("vllm_lora_load", f"{L}x{len(mods)}x2 strided H2D, rank {rank}/{max_rank}", steps=steps,
                    reset=reset, check=check, torch_fns={"fw": fw}, ce_fns={"ce_scope": ce_scope},
                    frozen="ce_scope", reps=reps, rounds=rounds)


def case_step_metadata(reps, rounds, R=256, max_reqs=1024, max_blocks=512, new_per_req=1):
    """Per-step input H2D (vLLM gpu_input_batch / block_table, SGLang ScheduleBatch): ~8 pinned
    metadata arrays plus a ragged per-request block-table append. vLLM v2 runner applies the
    block-table diff with one UVA Triton kernel; fw = the per-request torch fallback;
    alt = vectorized torch (H2D the diff, then index_put_). Requests and their rows change every
    step; the pinned staging arrays are refilled in place (what the runners do)."""
    meta_h = [pin(torch.zeros(R + 1 if i % 3 == 0 else R, dtype=torch.int32 if i % 2 else torch.int64))
              for i in range(8)]
    meta_d = [torch.empty_like(t, device=DEV) for t in meta_h]
    bt = torch.zeros(max_reqs, max_blocks, dtype=torch.int32, device=DEV)
    vals = pin(torch.zeros(R, new_per_req, dtype=torch.int32))
    flat = pin(torch.zeros(R * new_per_req, dtype=torch.int64))

    def make(r):
        rows = r.choice(max_reqs, R, replace=False)
        cols = r.integers(0, max_blocks - new_per_req, R)
        return {"rows": rows.tolist(), "cols": cols.tolist(),
                "meta": [torch.from_numpy(r.integers(0, 1 << 20, t.numel())).to(t.dtype) for t in meta_h],
                "vals": torch.from_numpy(r.integers(1, 1 << 20, (R, new_per_req)).astype(np.int32)),
                "flat": torch.from_numpy((rows[:, None] * max_blocks + cols[:, None] + np.arange(new_per_req))
                                         .reshape(-1))}
    steps = Steps(make)

    def stage(v):   # the runner fills its pinned staging buffers (not timed apart: cheap host writes)
        for t, x in zip(meta_h, v["meta"]):
            t.copy_(x)
        vals.copy_(v["vals"])
        flat.copy_(v["flat"])

    def reset(v):
        bt.zero_()
        for t in meta_d:
            t.zero_()

    def check(v):
        ref = torch.zeros(max_reqs, max_blocks, dtype=torch.int32)
        for i in range(R):
            ref[v["rows"][i], v["cols"][i]:v["cols"][i] + new_per_req] = v["vals"][i]
        exact(bt.cpu(), ref)
        for a, b in zip(meta_d, v["meta"]):
            exact(a.cpu(), b)

    def fw(v):
        stage(v)
        rl, cl = v["rows"], v["cols"]
        for a, b in zip(meta_d, meta_h):
            a.copy_(b, non_blocking=True)
        for i in range(R):
            bt[rl[i], cl[i]:cl[i] + new_per_req].copy_(vals[i], non_blocking=True)

    def vec(v):
        stage(v)
        for a, b in zip(meta_d, meta_h):
            a.copy_(b, non_blocking=True)
        bt.view(-1).index_put_((flat.to(DEV, non_blocking=True),), vals.view(-1).to(DEV, non_blocking=True))

    def ce_scope(e, v):
        with e.scope():
            fw(v)

    def step(e):
        def run(flat_h):
            e.copy_many(meta_d, meta_h)
            e.copy_(ce.lazy(bt.view(-1))[flat_h], ce.lazy(vals.view(-1)))
        return run

    return run_case("step_metadata_h2d", f"8 arrays + {R} block-table appends", steps=steps,
                    reset=reset, check=check, torch_fns={"fw": fw, "alt": vec}, ce_fns={"ce_scope": ce_scope},
                    frozen=None, plan=(step, lambda v: (stage(v), (v["flat"],))[1]),
                    reps=reps, rounds=rounds)


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

    def make(r):
        hid, did = paged_ids(n, page, HS, S, r)
        order = np.argsort(hid, kind="stable")    # sglang sorts host indices to lengthen runs
        hid, did = hid[order], did[order]
        return {"hid": hid, "did": did, "runs": runs_of((hid, did)), "hid_t": torch.from_numpy(hid),
                "did_d": torch.from_numpy(did).to(DEV)}
    steps = Steps(make)

    def reset(v):
        for l in range(L):
            kb[l][v["did_d"]] = 0
            vb[l][v["did_d"]] = 0

    def check(v):
        for l in range(L):
            exact(kb[l][v["did_d"]].cpu(), hk(0, l)[v["hid_t"]])
            exact(vb[l][v["did_d"]].cpu(), hk(1, l)[v["hid_t"]])

    def direct_layer(v, l):
        for s, d, m in v["runs"]:
            kb[l][d:d + m].copy_(hk(0, l)[s:s + m], non_blocking=True)
            vb[l][d:d + m].copy_(hk(1, l)[s:s + m], non_blocking=True)

    def index_layer(v, l):
        kb[l][v["did_d"]] = hk(0, l)[v["hid_t"]].to(DEV, non_blocking=True)
        vb[l][v["did_d"]] = hk(1, l)[v["hid_t"]].to(DEV, non_blocking=True)

    def fw(v):
        for l in range(L):
            (direct_layer if layout == "layer_first" else index_layer)(v, l)

    def alt(v):
        for l in range(L):
            (index_layer if layout == "layer_first" else direct_layer)(v, l)

    def ce_scope(e, v):
        for l in range(L):                    # one flush per layer, like the per-layer load
            with e.scope():
                direct_layer(v, l)

    row_el = H * D

    def ce_api(e, v):
        for l in range(L):
            with e.scope():
                for kv, buf in ((0, kb), (1, vb)):
                    x = hk(kv, l)   # a strided view: address the whole pool
                    e.segments(host, buf[l], x.storage_offset() + v["hid"] * x.stride(0), v["did"] * row_el,
                               [row_el] * n)

    def step(e):
        def run(hid_t, did_d):   # names the pools itself: a step plan checks the tensors it names
            for l in range(L):
                for kv, buf in ((0, kb), (1, vb)):
                    src = host[kv, l] if layout == "layer_first" else host[kv, :, l]
                    e.copy_(ce.lazy(buf[l])[did_d], ce.lazy(src)[hid_t])
        return run

    return run_case(f"sglang_hicache_load_{layout}", f"{L} layers x {n} tokens x K,V, page {page}", steps=steps,
                    reset=reset, check=check, torch_fns={"fw": fw, "alt": alt},
                    ce_fns={"ce_scope": ce_scope, "ce_api": ce_api}, frozen=None,
                    plan=(step, lambda v: (v["hid_t"], v["did_d"])), reps=reps, rounds=rounds)


def case_hicache_backup(reps, rounds, L=32, H=8, D=128, S=16384, HS=4096, n=2048, page=16):
    """SGLang HiCache L1 -> L2 backup_from_device_all_layer into the default page_first host
    layout [2, HS, L, H, D]: all layers at once, D2H. Default backend: sgl_kernel UVA kernel
    (staged through a GPU buffer for page_first); fw = the direct backend's run-merging copy_ loop."""
    kb = [torch.randn(S, H, D, dtype=BF16, device=DEV) for _ in range(L)]
    vb = [torch.randn(S, H, D, dtype=BF16, device=DEV) for _ in range(L)]
    host = pin(torch.zeros(2, HS, L, H, D, dtype=BF16))

    def make(r):
        did, hid = paged_ids(n, page, S, HS, r)
        order = np.argsort(did, kind="stable")
        did, hid = did[order], hid[order]
        return {"did": did, "hid": hid, "runs": runs_of((did, hid)), "hid_t": torch.from_numpy(hid),
                "did_d": torch.from_numpy(did).to(DEV)}
    steps = Steps(make)

    def reset(v):
        host[:, v["hid_t"]] = 0

    def check(v):
        for l in range(L):
            exact(host[0, v["hid_t"], l], kb[l][v["did_d"]].cpu())
            exact(host[1, v["hid_t"], l], vb[l][v["did_d"]].cpu())

    def fw(v):
        for l in range(L):
            for s, d, m in v["runs"]:
                host[0, d:d + m, l].copy_(kb[l][s:s + m], non_blocking=True)
                host[1, d:d + m, l].copy_(vb[l][s:s + m], non_blocking=True)

    def alt(v):
        for l in range(L):
            host[0, v["hid_t"], l] = kb[l][v["did_d"]].cpu()
            host[1, v["hid_t"], l] = vb[l][v["did_d"]].cpu()

    def ce_scope(e, v):
        with e.scope():
            fw(v)

    row_el = H * D

    def ce_api(e, v):
        with e.scope():
            for l in range(L):
                for kv, buf in ((0, kb), (1, vb)):
                    x = host[kv, :, l]
                    e.segments(buf[l], host, v["did"] * row_el, x.storage_offset() + v["hid"] * x.stride(0),
                               [row_el] * n)

    def step(e):
        def run(did_d, hid_t):
            for l in range(L):
                e.copy_(ce.lazy(host[0, :, l])[hid_t], ce.lazy(kb[l])[did_d])
                e.copy_(ce.lazy(host[1, :, l])[hid_t], ce.lazy(vb[l])[did_d])
        return run

    return run_case("sglang_hicache_backup_pf", f"{L} layers x {n} tokens x K,V -> page_first", steps=steps,
                    reset=reset, check=check, torch_fns={"fw": fw, "alt": alt},
                    ce_fns={"ce_scope": ce_scope, "ce_api": ce_api}, frozen="ce_api",
                    plan=(step, lambda v: (v["did_d"], v["hid_t"])), reps=reps, rounds=rounds)


def case_set_kv(reps, rounds, n=256, L=32, H=8, D=128, S=65536, quant=False):
    """SGLang MHATokenToKVPool.set_kv_buffer torch fallback: per layer k_cache[loc] = k (index_put
    by out_cache_loc, a GPU tensor). Frameworks use a store kernel (sglang store_cache / vLLM
    reshape_and_cache). quant=True stores int8 with a per-tensor scale (a stand-in for fp8 e4m3)."""
    dt = torch.int8 if quant else BF16
    kc = [torch.zeros(S, H, D, dtype=dt, device=DEV) for _ in range(L)]
    k = [torch.randn(n, H, D, dtype=BF16, device=DEV) * 3 for _ in range(L)]
    s = 0.05
    fn = ce.Elementwise(scale=1 / s, round=True, clamp=(-127, 127)) if quant else None
    steps = Steps(lambda r: {"loc_d": torch.from_numpy(r.choice(S, n, replace=False).astype(np.int64)).to(DEV)})

    def q(x):
        return torch.clamp(torch.round(x.float() * (1 / s)), -127, 127).to(torch.int8) if quant else x

    ref = [q(x) for x in k]

    def reset(v):
        for c in kc:
            c[v["loc_d"]] = 0

    def check(v):
        for c, x in zip(kc, ref):
            exact(c[v["loc_d"]], x)

    def fw(v):
        for c, x in zip(kc, k):
            c[v["loc_d"]] = q(x)

    def ce_api(e, v):
        for c, x in zip(kc, k):              # one store per layer, between that layer's kernels
            e.scatter_(c, v["loc_d"], x, fn=fn)

    def step(e):
        def run(loc_d):
            for c, x in zip(kc, k):
                e.scatter_(c, loc_d, x, fn=fn)
        return run

    tag = "int8" if quant else "bf16"
    return run_case(f"set_kv_{tag}_n{n}", f"{L} layers, scatter {n} tokens by out_cache_loc", steps=steps,
                    reset=reset, check=check, torch_fns={"fw": fw}, ce_fns={"ce_api": ce_api},
                    frozen=None, plan=(step, lambda v: (v["loc_d"],)), reps=reps, rounds=rounds)


def case_move_kv(reps, rounds, n=64, L=32, H=8, D=128, S=65536):
    """SGLang move_kv_cache (spec-decode accept / compaction) torch path: per layer
    k[tgt] = k[src]; v[tgt] = v[src] (GPU index tensors). CUDA default: one Triton launch over a
    pointer table."""
    kb = [torch.randn(S, H, D, dtype=BF16, device=DEV) for _ in range(L)]
    vb = [torch.randn(S, H, D, dtype=BF16, device=DEV) for _ in range(L)]

    def make(r):
        sel = r.choice(S, 2 * n, replace=False).astype(np.int64)
        return {"src_d": torch.from_numpy(sel[:n]).to(DEV), "tgt_d": torch.from_numpy(sel[n:]).to(DEV)}
    steps = Steps(make)
    want = {}

    def reset(v):
        want["ref"] = [(k_[v["src_d"]].clone(), v_[v["src_d"]].clone()) for k_, v_ in zip(kb, vb)]
        for k_, v_ in zip(kb, vb):
            k_[v["tgt_d"]] = 0
            v_[v["tgt_d"]] = 0

    def check(v):
        for (k_, v_), (rk, rv) in zip(zip(kb, vb), want["ref"]):
            exact(k_[v["tgt_d"]], rk)
            exact(v_[v["tgt_d"]], rv)

    def fw(v):
        for k_, v_ in zip(kb, vb):
            k_[v["tgt_d"]] = k_[v["src_d"]]
            v_[v["tgt_d"]] = v_[v["src_d"]]

    def ce_api(e, v):
        with e.scope():
            for k_, v_ in zip(kb, vb):
                e.copy_(ce.lazy(k_)[v["tgt_d"]], ce.lazy(k_)[v["src_d"]])
                e.copy_(ce.lazy(v_)[v["tgt_d"]], ce.lazy(v_)[v["src_d"]])

    def step(e):
        def run(src_d, tgt_d):
            for k_, v_ in zip(kb, vb):
                e.copy_(ce.lazy(k_)[tgt_d], ce.lazy(k_)[src_d])
                e.copy_(ce.lazy(v_)[tgt_d], ce.lazy(v_)[src_d])
        return run

    return run_case(f"sglang_move_kv_n{n}", f"{L} layers x K,V, {n} token slots D2D", steps=steps,
                    reset=reset, check=check, torch_fns={"fw": fw}, ce_fns={"ce_api": ce_api},
                    frozen="ce_api", plan=(step, lambda v: (v["src_d"], v["tgt_d"])), reps=reps, rounds=rounds)


def case_pd_staging(reps, rounds, L=32, H=8, D=128, S=16384, n=4096, nh=4, h0=4):
    """SGLang PD disaggregation with heterogeneous TP (staging_buffer._gather_all_layers_torch):
    gather a head slice of scattered tokens of every layer into one contiguous staging buffer
    for a single RDMA (GPU token indices). Default is a Triton kernel; fw = the torch path."""
    kb = [torch.randn(S, H, D, dtype=BF16, device=DEV) for _ in range(L)]
    vb = [torch.randn(S, H, D, dtype=BF16, device=DEV) for _ in range(L)]
    stg = torch.empty(2, L, n, nh, D, dtype=BF16, device=DEV)

    def make(r):
        idx_d = torch.from_numpy(r.choice(S, n, replace=False).astype(np.int64)).to(DEV)
        return {"idx_d": idx_d, "gidx": idx_d.view(-1, 1, 1).expand(n, nh, D)}
    steps = Steps(make)

    def reset(v):
        stg.zero_()

    def check(v):
        for l in range(L):
            exact(stg[0, l], kb[l][v["idx_d"], h0:h0 + nh])
            exact(stg[1, l], vb[l][v["idx_d"], h0:h0 + nh])

    def fw(v):
        for kv, buf in ((0, kb), (1, vb)):
            for l in range(L):
                torch.gather(buf[l][:, h0:h0 + nh], 0, v["gidx"], out=stg[kv, l])

    def ce_api(e, v):
        with e.scope():
            for kv, buf in ((0, kb), (1, vb)):
                for l in range(L):
                    e.copy_(stg[kv, l], ce.lazy(buf[l])[v["idx_d"]].slice(1, h0, h0 + nh))

    def step(e):
        def run(idx_d):
            for kv, buf in ((0, kb), (1, vb)):
                for l in range(L):
                    e.copy_(stg[kv, l], ce.lazy(buf[l])[idx_d].slice(1, h0, h0 + nh))
        return run

    return run_case("sglang_pd_staging_gather", f"{L} layers x K,V, {n} tokens, heads {h0}:{h0 + nh}", steps=steps,
                    reset=reset, check=check, torch_fns={"fw": fw}, ce_fns={"ce_api": ce_api},
                    frozen="ce_api", plan=(step, lambda v: (v["idx_d"],)), reps=reps, rounds=rounds)


def case_mamba_cow(reps, rounds, L=24, slots=17, HV=32, V=128, K=128, conv_dim=8192, kc=3, n=4):
    """SGLang MambaPool.copy_from (prefix-cache copy-on-write of hybrid-model state), all layers
    at once: conv[:, dst] = conv[:, src]; temporal[:, dst] = temporal[:, src] (Qwen3-Next-like GDN)."""
    conv = torch.randn(L, slots, conv_dim, kc, dtype=BF16, device=DEV)
    temp = torch.randn(L, slots, HV, V, K, dtype=torch.float32, device=DEV)

    def make(r):
        sel = r.choice(slots, 2 * n, replace=False).astype(np.int64)
        return {"src_d": torch.from_numpy(sel[:n]).to(DEV), "dst_d": torch.from_numpy(sel[n:]).to(DEV)}
    steps = Steps(make)
    want = {}

    def reset(v):
        want["rc"], want["rt"] = conv[:, v["src_d"]].clone(), temp[:, v["src_d"]].clone()
        conv[:, v["dst_d"]] = 0
        temp[:, v["dst_d"]] = 0

    def check(v):
        exact(conv[:, v["dst_d"]], want["rc"])
        exact(temp[:, v["dst_d"]], want["rt"])

    def fw(v):
        conv[:, v["dst_d"]] = conv[:, v["src_d"]]
        temp[:, v["dst_d"]] = temp[:, v["src_d"]]

    def ce_api(e, v):   # torch's conv[:, dst] = conv[:, src] (gathers first: one pool on both sides)
        with e.scope():
            e.copy_(ce.lazy(conv).index_select(1, v["dst_d"]), ce.lazy(conv).index_select(1, v["src_d"]))
            e.copy_(ce.lazy(temp).index_select(1, v["dst_d"]), ce.lazy(temp).index_select(1, v["src_d"]))

    def step(e):
        def run(src_d, dst_d):
            e.copy_(ce.lazy(conv).index_select(1, dst_d), ce.lazy(conv).index_select(1, src_d))
            e.copy_(ce.lazy(temp).index_select(1, dst_d), ce.lazy(temp).index_select(1, src_d))
        return run

    return run_case("sglang_mamba_cow", f"{L} layers, {n} slots, {HV * V * K * 4 >> 20} MB/slot/layer", steps=steps,
                    reset=reset, check=check, torch_fns={"fw": fw}, ce_fns={"ce_api": ce_api},
                    frozen="ce_api", plan=(step, lambda v: (v["src_d"], v["dst_d"])), reps=reps, rounds=rounds)


def case_moe_lora(reps, rounds, L=16, E=64, max_loras=2, max_rank=32, rank=16, hidden=2048, inter=768):
    """SGLang LoRA memory pool, MoE adapter (load_lora_weight_to_buffer / copy_weight_into_buffer):
    per (layer, expert, module) strided H2D copies of pinned weights into slot `buffer_id`, which
    changes from load to load."""
    shapes = {"gate_A": (max_rank, hidden), "up_A": (max_rank, hidden), "down_A": (max_rank, inter),
              "gate_B": (inter, max_rank), "up_B": (inter, max_rank), "down_B": (hidden, max_rank)}
    buf = {(l, m): torch.zeros(max_loras, E, *s, dtype=BF16, device=DEV) for l in range(L)
           for m, s in shapes.items()}
    src = {}
    for (l, m), b in buf.items():
        r_, c_ = b.shape[2:]
        src[(l, m)] = pin(torch.randn(E, *((rank, c_) if m.endswith("A") else (r_, rank)), dtype=BF16))
    steps = Steps(lambda r: {"slot": int(r.integers(0, max_loras))})

    def view_of(slot, l, m, e):
        b = buf[(l, m)][slot, e]
        return b[:rank] if m.endswith("A") else b[:, :rank]

    def reset(v):
        for b in buf.values():
            b[v["slot"]].zero_()

    def check(v):
        for (l, m), s in src.items():
            b = buf[(l, m)][v["slot"]]
            got = b[:, :rank] if m.endswith("A") else b[:, :, :rank]
            exact(got, s.to(DEV))

    def fw(v):
        for (l, m), s in src.items():
            for e in range(E):
                view_of(v["slot"], l, m, e).copy_(s[e], non_blocking=True)

    def ce_scope(e_, v):
        with e_.scope():
            fw(v)

    return run_case("sglang_moe_lora_load", f"{L}x{E}x{len(shapes)} strided H2D, rank {rank}", steps=steps,
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

COLS = ["fw", "alt", "ce_scope", "ce_api", "ce_plan", "ce_frozen"]


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
        ce_keys = [c for c in ("ce_scope", "ce_api", "ce_plan") if c in us]  # ce_frozen: fixed ids only
        best = min(ce_keys, key=lambda c: us[c])
        strat = f"{','.join(r['strategies'][best])} ({r['calls'][best]})"
        notes = "".join(f"  [{k}: {n}]" for k, n in r.get("notes", {}).items())
        print(f"{r['case']:34s}{cells}  {us['fw'] / us[best]:6.2f}x  {strat}{notes}", flush=True)
        rows.append(r)
    if args.json:
        out = {"hardware": ce.hardware_fingerprint(), "torch": torch.__version__, "copyengine": ce.__version__,
               "unit": "us", "rows": rows}
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
