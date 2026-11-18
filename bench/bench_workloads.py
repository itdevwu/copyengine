#!/usr/bin/env python3
"""copyengine on the workloads it targets, one suite per kind, one JSON report.

Covers the capability matrix across versions (a suite that an older copyengine cannot run is
reported as skipped), so two reports compare release against release (--compare):
  1. probe            : Hardware roofline baselines (H2D/D2H pinned/pageable, D2D)
  2. sweep            : Log2 segment-size sweep (512 B -> 4 MiB, constant stride) vs PyTorch / compile
  3. paged_kv         : 56-layer heterogeneous GQA paged KV H2D/D2H (pinned & pageable) with dynamic IDs;
                        torch and copyengine run the same task in every pairing
  4. reblock          : 56-layer cross-axis reblock+transpose scatter (block-aligned StepPlan vs unaligned miss)
  5. reshard          : TP2->TP4 resharding, 24 layers: 96 tensor pairs, 1536 head slices; checked
                        bit for bit; vs a copy_ loop, _foreach_copy_ and the loop captured in a CUDA graph
  6. tma_sm_cap       : Pure D2D & narrow-row transpose (`bulk` & `tma2d`, sm90+) across SM caps (0, 8, 4)
  7. interference     : Foreground GEMM slowdown while a background transfer runs for the whole
                        foreground window; background throughput inside that window (Pareto points)
  8. tile_quant       : 24-layer GPU paged gather + quant/dequant + transposed V-cache + bits vs compile
  9. pcie_quant       : Cross-PCIe quantizing offload/onload (1 tensor & 24 layers, paged GPU->host pool)
 10. box_and_compose  : (128,128) block FP8 weight quant, StepPlan temp elimination, torch.compile hybrid
 11. quant_formats    : every quantization format (per group, packed, MX, NVFP4, block 128x128),
                        quantize & dequantize vs the torch recipe eager and compiled, bits vs the recipe
 12. calls            : small-copy call overhead: gathers of 1-64 blocks (explicit, step plan, torch,
                        CUDA graph) and one 32 KB copy per call split into view, dispatch and queue
 13. p2p              : two GPUs: KV migration, TP resharding, converting migration (skipped on one GPU)

Timing: `measure_stream` returns (wall_us, event_us), both medians. Each repetition starts from an
idle GPU (synchronize, then an L2 flush unless --no-flush-l2), so the event interval also contains
the host's planning and submission: both are end-to-end times, not pure device times.

Host-side torch baselines copy straight from pinned memory (per block, `_foreach_copy_`), never
through a CPU-side gather into pageable memory.

Runs anywhere from sm_75 (Colab T4) up: what the GPU, torch or copyengine version cannot do is
reported as skipped, a failing suite is recorded and the next one runs.

Results are machine specific; keep them out of the repository.

Usage:
  python bench/bench_workloads.py --json report.json
  python bench/bench_workloads.py --suites probe,sweep,tile_quant --quick
  python bench/bench_workloads.py --compare report_old.json report_new.json
"""

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import torch._dynamo as dynamo
except Exception:
    dynamo = None

try:
    import einops
except ImportError:
    einops = None

import copyengine as ce

DEV = "cuda:0"
# Standard open-source-style GQA head layouts (per-TP-rank KV heads)
HETEROGENEOUS_56L_HEADS = [2] * 12 + [1] * 44
HETEROGENEOUS_24L_HEADS = [4, 2, 4, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 4]


# ---------------------------------------------------------------------------
# Capabilities & Timing Utilities
# ---------------------------------------------------------------------------

def detect_capabilities() -> Dict[str, Any]:
    probe = ce.probe(bandwidth=False)
    e = ce.Engine()
    a = torch.ones((256, 256), dtype=torch.float32, device=DEV)
    b = torch.empty_like(a)
    e.copy_(b, a)
    e.synchronize()
    st = e.stats()
    cuda_info = st.get("cuda", [{}])[0] if isinstance(st.get("cuda"), list) and st.get("cuda") else {}
    q_sig = inspect.signature(e.quantize).parameters if hasattr(e, "quantize") else {}
    return {
        "copyengine_version": getattr(ce, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "shim_loaded": bool(getattr(getattr(ce, "_shim", None), "lib", None) is not None),
        "einops_version": getattr(einops, "__version__", None),
        "has_copy_many": hasattr(ce, "copy_many"),
        "has_plan": hasattr(ce, "plan"),
        "has_quantize": hasattr(ce, "quantize"),
        "has_quant_block": "block" in q_sig,
        "has_compiled_stats": hasattr(ce, "compiled_stats"),
        "cuda_bulk": cuda_info.get("bulk"),
        "cuda_tma2d": cuda_info.get("tma2d"),
        "cuda_sm_limit": cuda_info.get("sm_limit"),
        "cuda_batch": cuda_info.get("batch"),
        "bf16": bf16_ok(),
        "has_fp8": has_dtype("float8_e4m3fn"),
        "has_e8m0": has_dtype("float8_e8m0fnu"),
        "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "probe": probe,
    }


FLUSH_L2 = True
_L2_BUF: Optional[torch.Tensor] = None


def flush_l2() -> None:
    """Evict the inputs of the previous repetition from L2 (outside the timed interval)."""
    global _L2_BUF
    if not FLUSH_L2:
        return
    if _L2_BUF is None:
        l2 = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0) or (128 << 20)
        _L2_BUF = torch.empty(2 * l2, dtype=torch.uint8, device=DEV)
    _L2_BUF.random_()


def cc() -> Tuple[int, int]:
    return torch.cuda.get_device_capability(0)


def has_dtype(name: str) -> bool:
    return hasattr(torch, name)


def bf16_ok() -> bool:
    return torch.cuda.is_bf16_supported()


def short_err(e: BaseException) -> str:
    lines = str(e).strip().splitlines()
    return f"{type(e).__name__}: {lines[0][:160] if lines else ''}"


def measure_stream(
    fn: Callable[[], Any],
    stream: torch.cuda.Stream,
    *,
    warmup: int = 4,
    reps: int = 15,
) -> Tuple[float, float]:
    """Return (median_wall_us, median_event_us) on an explicit CUDA stream.

    Every repetition starts from an idle GPU, so the event interval includes the host's planning
    and submission as well: both numbers are end-to-end.
    """
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    wall_us, gpu_us = [], []
    for _ in range(reps):
        flush_l2()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            t0 = time.perf_counter()
            e0.record(stream)
            fn()
            e1.record(stream)
            stream.synchronize()
            t1 = time.perf_counter()
        wall_us.append(round((t1 - t0) * 1e6, 1))
        gpu_us.append(round(e0.elapsed_time(e1) * 1e3, 1))
    return statistics.median(wall_us), statistics.median(gpu_us)


def cleanup_cuda() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def reset_dynamo() -> None:
    if dynamo is not None:
        dynamo.reset()


def get_quant_ref():
    try:
        return importlib.import_module("copyengine._quant")
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Suite 1: Hardware Roofline Probe
# ---------------------------------------------------------------------------

def suite_probe(stream: torch.cuda.Stream, reps: int, quick: bool, **_) -> Dict[str, Any]:
    nbytes = 64 * 1024 * 1024
    gpu_src = torch.randint(1, 120, (nbytes,), dtype=torch.uint8, device=DEV)
    gpu_dst = torch.empty_like(gpu_src)
    pin_src = torch.randint(1, 120, (nbytes,), dtype=torch.uint8).pin_memory()
    pin_dst = torch.empty_like(pin_src).pin_memory()
    pag_src = torch.randint(1, 120, (nbytes,), dtype=torch.uint8)
    pag_dst = torch.empty_like(pag_src)

    def gbps(us: float) -> float:
        return round(nbytes / us / 1e3, 2)

    h2d_pin = measure_stream(lambda: gpu_dst.copy_(pin_src, non_blocking=True), stream, reps=reps)[1]
    d2h_pin = measure_stream(lambda: pin_dst.copy_(gpu_src, non_blocking=True), stream, reps=reps)[1]
    h2d_pag = measure_stream(lambda: gpu_dst.copy_(pag_src, non_blocking=True), stream, reps=max(5, reps // 2))[1]
    d2h_pag = measure_stream(lambda: pag_dst.copy_(gpu_src, non_blocking=True), stream, reps=max(5, reps // 2))[1]
    d2d = measure_stream(lambda: gpu_dst.copy_(gpu_src, non_blocking=True), stream, reps=reps)[1]

    del gpu_src, gpu_dst, pin_src, pin_dst, pag_src, pag_dst
    cleanup_cuda()
    return {
        "bytes": nbytes,
        "h2d_pinned_us": h2d_pin,
        "h2d_pinned_gbps": gbps(h2d_pin),
        "d2h_pinned_us": d2h_pin,
        "d2h_pinned_gbps": gbps(d2h_pin),
        "h2d_pageable_us": h2d_pag,
        "h2d_pageable_gbps": gbps(h2d_pag),
        "d2h_pageable_us": d2h_pag,
        "d2h_pageable_gbps": gbps(d2h_pag),
        "d2d_us": d2d,
        "d2d_gbps": gbps(d2d),
    }


# ---------------------------------------------------------------------------
# Suite 2: Log2 Segment-Size Bandwidth Sweep (Constant Stride, 64 MiB Pinned H2D)
# ---------------------------------------------------------------------------

def suite_sweep(
    stream: torch.cuda.Stream,
    reps: int,
    quick: bool,
    with_compile: bool,
    rooflines: Dict[str, Any],
    **_,
) -> Dict[str, Any]:
    total = 64 * 1024 * 1024
    # Constant stride (every 2nd slot -> guaranteed non-contiguous without random clustering)
    pool_bytes = 128 * 1024 * 1024
    host_pool = torch.randint(1, 120, (pool_bytes,), dtype=torch.uint8).pin_memory()
    dev_pool = torch.zeros((pool_bytes,), dtype=torch.uint8, device=DEV)
    ceil_gbps = rooflines["h2d_pinned_gbps"]

    sizes = [512, 2048, 8192, 32768, 131072, 524288, 2097152] if quick else [
        512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304
    ]

    rows = []
    for seg in sizes:
        reset_dynamo()
        n_segs = total // seg
        max_rows = pool_bytes // seg
        # Constant stride=2 permutation with reversed destination order to prevent range coalescing
        h_idx_np = np.arange(0, 2 * n_segs, 2, dtype=np.int64)
        d_idx_np = h_idx_np[::-1].copy()
        h_idx_alt = h_idx_np.copy()
        d_idx_alt = np.roll(d_idx_np, 1)

        h_idx_cpu = torch.from_numpy(h_idx_np)
        d_idx_gpu = torch.from_numpy(d_idx_np).to(DEV)
        h_2d = host_pool[: max_rows * seg].view(max_rows, seg)
        d_2d = dev_pool[: max_rows * seg].view(max_rows, seg)
        # Gather on the CPU into a pinned staging buffer (not pageable memory), one H2D, scatter on the GPU.
        pin_stage = torch.empty((n_segs, seg), dtype=torch.uint8).pin_memory()
        gpu_stage = torch.empty((n_segs, seg), dtype=torch.uint8, device=DEV)

        def torch_vec():
            torch.index_select(h_2d, 0, h_idx_cpu, out=pin_stage)
            gpu_stage.copy_(pin_stage, non_blocking=True)
            d_2d.index_copy_(0, d_idx_gpu, gpu_stage)

        vec_us = measure_stream(torch_vec, stream, warmup=2, reps=max(5, reps // 2))[1]

        foreach_us = None
        if n_segs <= 8192:
            src_list = [h_2d[i] for i in h_idx_np.tolist()]
            dst_list = [d_2d[j] for j in d_idx_np.tolist()]
            foreach_us = measure_stream(
                lambda: torch._foreach_copy_(dst_list, src_list, non_blocking=True),
                stream,
                warmup=2,
                reps=max(5, reps // 2),
            )[1]

        comp_us = None
        if with_compile and dynamo is not None:
            try:
                c_vec = torch.compile(lambda d, h, di, hi: d.index_copy_(0, di, h.index_select(0, hi).to(DEV, non_blocking=True)))
                with torch.cuda.stream(stream):
                    c_vec(d_2d, h_2d, d_idx_gpu, h_idx_cpu)
                torch.cuda.synchronize()
                comp_us = measure_stream(lambda: c_vec(d_2d, h_2d, d_idx_gpu, h_idx_cpu), stream, warmup=2, reps=max(5, reps // 2))[1]
            except Exception:
                comp_us = None

        eng = ce.Engine()
        if hasattr(ce, "plan"):
            def ce_step(d, h, di, hi):
                eng.copy_(ce.lazy(d).index_select(0, di), ce.lazy(h).index_select(0, hi))

            plan_ce = eng.plan(ce_step)
            with torch.cuda.stream(stream):
                for k in range(75):
                    plan_ce(d_2d, h_2d, d_idx_np if k % 2 == 0 else d_idx_alt, h_idx_np if k % 2 == 0 else h_idx_alt)
            torch.cuda.synchronize()
            ce_us = measure_stream(lambda: plan_ce(d_2d, h_2d, d_idx_np, h_idx_np), stream, warmup=3, reps=reps)[1]
        else:
            with torch.cuda.stream(stream):
                for _ in range(75):
                    eng.copy_(ce.lazy(d_2d).index_select(0, d_idx_np), ce.lazy(h_2d).index_select(0, h_idx_np))
            torch.cuda.synchronize()
            ce_us = measure_stream(eng.freeze().run, stream, warmup=3, reps=reps)[1]

        strat = eng.stats()["last"]["decisions"][0]["strategy"]

        def to_pct(us: Optional[float]) -> Optional[float]:
            if us is None or us <= 0:
                return None
            return round((total / us / 1e3) / ceil_gbps * 100.0, 1)

        rows.append({
            "seg_bytes": seg,
            "n_segs": n_segs,
            "torch_vec_us": vec_us,
            "torch_vec_pct": to_pct(vec_us),
            "foreach_us": foreach_us,
            "foreach_pct": to_pct(foreach_us),
            "compile_us": comp_us,
            "compile_pct": to_pct(comp_us),
            "ce_us": ce_us,
            "ce_pct": to_pct(ce_us),
            "ce_strategy": strat,
        })

    del host_pool, dev_pool
    cleanup_cuda()
    return {"total_bytes": total, "pcie_ceiling_gbps": ceil_gbps, "rows": rows}


# ---------------------------------------------------------------------------
# Suite 3: 56-Layer Heterogeneous Paged KV H2D/D2H (Dynamic IDs & Pageable Bounce)
# ---------------------------------------------------------------------------

def suite_paged_kv(stream: torch.cuda.Stream, reps: int, quick: bool, **_) -> Dict[str, Any]:
    b_dev, b_host, S, D = 128, 128, 512, 128
    heads = HETEROGENEOUS_56L_HEADS
    dev_layers = [torch.randint(1, 120, (b_dev, S, h, 2, D), dtype=torch.uint8, device=DEV) for h in heads]
    blk_bytes = sum(t[0].nbytes for t in dev_layers)

    pin_pool = torch.randint(1, 120, (b_host, blk_bytes), dtype=torch.uint8).pin_memory()
    pag_pool = torch.randint(1, 120, (b_host, blk_bytes), dtype=torch.uint8)

    def make_views(pool: torch.Tensor):
        hs, ds = [], []
        off = 0
        for t in dev_layers:
            lb = t[0].nbytes
            hs.append(pool[:, off : off + lb])
            ds.append(t.reshape(b_dev, lb))
            off += lb
        return hs, ds

    pin_views, dev_views = make_views(pin_pool)
    pag_views, _ = make_views(pag_pool)

    block_counts = [16] if quick else [8, 16, 32]
    results = []
    for nb in block_counts:
        n_perms = 60
        h_perms, d_perms = [], []
        for k in range(n_perms):
            rng = np.random.default_rng(4000 + nb * 100 + k)
            h_perms.append(rng.permutation(b_host)[:nb].astype(np.int64))
            d_perms.append(rng.permutation(b_dev)[:nb].astype(np.int64))

        payload_mb = round(nb * blk_bytes / 1e6, 2)
        for kind_name, h_views in [("pinned", pin_views), ("pageable", pag_views)]:
            for direction in ("H2D", "D2H"):
                h2d = direction == "H2D"
                # 1. PyTorch _foreach_copy_ baseline on dynamic per-layer views
                idx_p = [0]
                def torch_foreach_dyn():
                    k = idx_p[0] % n_perms; idx_p[0] += 1
                    hi, di = h_perms[k].tolist(), d_perms[k].tolist()
                    srcs, dsts = [], []
                    for hv, dv in zip(h_views, dev_views):
                        for h_i, d_i in zip(hi, di):
                            srcs.append(hv[h_i] if h2d else dv[d_i])
                            dsts.append(dv[d_i] if h2d else hv[h_i])
                    torch._foreach_copy_(dsts, srcs, non_blocking=True)

                foreach_us = measure_stream(torch_foreach_dyn, stream, warmup=2, reps=max(5, reps // 2))

                # 2. Eager ce.scope() on dynamic IDs (shows Plan-Miss cost when indices are NOT parameterized)
                eng_miss = ce.Engine()
                with torch.cuda.stream(stream):
                    for _ in range(30):
                        with eng_miss.scope():
                            for hv, dv in zip(h_views, dev_views):
                                if h2d:
                                    eng_miss.copy_(ce.lazy(dv).index_select(0, d_perms[0]), ce.lazy(hv).index_select(0, h_perms[0]))
                                else:
                                    eng_miss.copy_(ce.lazy(hv).index_select(0, h_perms[0]), ce.lazy(dv).index_select(0, d_perms[0]))
                torch.cuda.synchronize()

                def ce_eager_miss():
                    k = idx_p[0] % n_perms; idx_p[0] += 1
                    hi, di = h_perms[k], d_perms[k]
                    with eng_miss.scope():
                        for hv, dv in zip(h_views, dev_views):
                            if h2d:
                                eng_miss.copy_(ce.lazy(dv).index_select(0, di), ce.lazy(hv).index_select(0, hi))
                            else:
                                eng_miss.copy_(ce.lazy(hv).index_select(0, hi), ce.lazy(dv).index_select(0, di))

                miss_us = measure_stream(ce_eager_miss, stream, warmup=1, reps=max(4, reps // 3))

                # 3. StepPlan with dynamic host indices (zero re-plan!)
                step_us = None
                step_strat = None
                d2d_parity_ok = None
                if hasattr(ce, "plan"):
                    try:
                        eng_sp = ce.Engine()
                        def step_fn(dvs, hvs, dix, hix):
                            with eng_sp.scope():
                                for hv, dv in zip(hvs, dvs):
                                    if h2d:
                                        eng_sp.copy_(ce.lazy(dv).index_select(0, dix), ce.lazy(hv).index_select(0, hix))
                                    else:
                                        eng_sp.copy_(ce.lazy(hv).index_select(0, hix), ce.lazy(dv).index_select(0, dix))
                        planned = eng_sp.plan(step_fn)
                        with torch.cuda.stream(stream):
                            for k in range(75):
                                planned(dev_views, h_views, d_perms[k % n_perms], h_perms[k % n_perms])
                        torch.cuda.synchronize()

                        # Verify bit-exact parity on unseen permutation
                        test_k = n_perms - 1
                        d2d_parity_ok = None
                        if h2d:
                            with torch.cuda.stream(stream):
                                for t in dev_layers:
                                    t.zero_()
                                for hv, dv in zip(h_views, dev_views):
                                    for h_i, d_i in zip(h_perms[test_k].tolist(), d_perms[test_k].tolist()):
                                        dv[d_i].copy_(hv[h_i], non_blocking=True)
                            torch.cuda.synchronize()
                            refs = [t.clone() for t in dev_layers]
                            with torch.cuda.stream(stream):
                                for t in dev_layers:
                                    t.zero_()
                                planned(dev_views, h_views, d_perms[test_k], h_perms[test_k])
                            torch.cuda.synchronize()
                            d2d_parity_ok = all(torch.equal(a, b) for a, b in zip(dev_layers, refs))

                        def run_sp():
                            k = idx_p[0] % n_perms; idx_p[0] += 1
                            planned(dev_views, h_views, d_perms[k], h_perms[k])
                        step_us = measure_stream(run_sp, stream, reps=reps)
                        decs = eng_sp.stats()["last"]["decisions"]
                        step_strat = decs[0]["strategy"] if decs else "none"
                    except Exception as e:
                        step_strat = f"unsupported:{type(e).__name__}"

                # 4. Contiguous prefix slices (one segment per layer): ce.copy_many vs torch on the same task
                copy_many_us = None
                copy_many_strat = None
                dst_sl = [dv[:nb] if h2d else hv[:nb] for hv, dv in zip(h_views, dev_views)]
                src_sl = [hv[:nb] if h2d else dv[:nb] for hv, dv in zip(h_views, dev_views)]
                foreach_prefix_us = measure_stream(
                    lambda: torch._foreach_copy_(dst_sl, src_sl, non_blocking=True), stream, reps=reps)
                if hasattr(ce, "copy_many"):
                    eng_cm = ce.Engine()
                    with torch.cuda.stream(stream):
                        for _ in range(60):
                            eng_cm.copy_many(dst_sl, src_sl)
                    torch.cuda.synchronize()
                    copy_many_us = measure_stream(lambda: eng_cm.copy_many(dst_sl, src_sl), stream, reps=reps)
                    cm_decs = eng_cm.stats()["last"]["decisions"]
                    copy_many_strat = cm_decs[0]["strategy"] if cm_decs else "none"

                results.append({
                    "host_kind": kind_name,
                    "direction": direction,
                    "blocks": nb,
                    "payload_mb": payload_mb,
                    # dynamic block IDs: torch_foreach_dyn_us vs ce_scope_dyn_miss_us / ce_step_plan_dyn_us
                    "torch_foreach_dyn_us": foreach_us,
                    "ce_scope_dyn_miss_us": miss_us,
                    "ce_step_plan_dyn_us": step_us,
                    "step_plan_strategy": step_strat,
                    # contiguous prefixes: torch_foreach_prefix_us vs ce_copy_many_us
                    "torch_foreach_prefix_us": foreach_prefix_us,
                    "ce_copy_many_us": copy_many_us,
                    "copy_many_strategy": copy_many_strat,
                    "parity_ok": d2d_parity_ok if (h2d and hasattr(ce, "plan")) else None,
                })

    del dev_layers, pin_pool, pag_pool
    cleanup_cuda()
    return {"layers": len(heads), "results": results}


# ---------------------------------------------------------------------------
# Suite 4: 56-Layer Cross-Axis Reblock & Transpose Scatter (Aligned, Unaligned, einops)
# ---------------------------------------------------------------------------

def suite_reblock(stream: torch.cuda.Stream, reps: int, quick: bool, with_compile: bool, **_) -> Dict[str, Any]:
    S_raw, S_buf, D = 512, 1024, 128
    b_raw, b_buf = 64, 32
    heads = HETEROGENEOUS_56L_HEADS
    raw_layers = [torch.randint(1, 120, (b_raw, S_raw, h, 2, D), dtype=torch.uint8, device=DEV) for h in heads]
    buf_dev = [torch.randint(1, 120, (b_buf, h, S_buf, 2, D), dtype=torch.uint8, device=DEV) for h in heads]
    buf_pin = [torch.randint(1, 120, (b_buf, h, S_buf, 2, D), dtype=torch.uint8).pin_memory() for h in heads]

    # Case 4A: Block-aligned dynamic block IDs (8 buf blocks of 1024 -> 16 raw blocks of 512, 142.6 MB)
    reqs = []
    for k in range(30):
        rng = np.random.default_rng(700 + k)
        b8 = rng.permutation(b_buf)[:8].astype(np.int64)
        r16 = rng.permutation(b_raw)[:16].astype(np.int64)
        re_np, ro_np = r16[0:16:2], r16[1:16:2]
        reqs.append((
            b8, re_np, ro_np,
            torch.from_numpy(b8).to(DEV), torch.from_numpy(re_np).to(DEV), torch.from_numpy(ro_np).to(DEV),
        ))

    def torch_d2d_56l(bs, rs, b_ix, re_ix, ro_ix):
        for sb_t, rb_t in zip(bs, rs):
            p = sb_t.index_select(0, b_ix)
            rb_t.index_copy_(0, re_ix, p[:, :, :512].permute(0, 2, 1, 3, 4))
            rb_t.index_copy_(0, ro_ix, p[:, :, 512:].permute(0, 2, 1, 3, 4))

    # Blocks straight from pinned memory into a GPU staging buffer (no CPU-side gather), then the permute on the GPU.
    stage_h2d = [torch.empty((8,) + tuple(t.shape[1:]), dtype=t.dtype, device=DEV) for t in buf_pin]

    def torch_h2d_56l(bs, rs, b_np, re_gpu, ro_gpu):
        ids = b_np.tolist()
        for sb_t, rb_t, p in zip(bs, rs, stage_h2d):
            torch._foreach_copy_([p[j] for j in range(len(ids))], [sb_t[i] for i in ids], non_blocking=True)
            rb_t.index_copy_(0, re_gpu, p[:, :, :512].permute(0, 2, 1, 3, 4))
            rb_t.index_copy_(0, ro_gpu, p[:, :, 512:].permute(0, 2, 1, 3, 4))

    it = [0]
    def run_torch_d2d():
        k = it[0] % len(reqs); it[0] += 1
        torch_d2d_56l(buf_dev, raw_layers, reqs[k][3], reqs[k][4], reqs[k][5])

    def run_torch_h2d():
        k = it[0] % len(reqs); it[0] += 1
        torch_h2d_56l(buf_pin, raw_layers, reqs[k][0], reqs[k][4], reqs[k][5])

    eager_d2d_us = measure_stream(run_torch_d2d, stream, reps=reps)
    eager_h2d_us = measure_stream(run_torch_h2d, stream, reps=max(5, reps // 2))

    compile_d2d_us = None
    compile_cg_d2d_us = None
    if with_compile and dynamo is not None:
        reset_dynamo()
        try:
            c_d2d = torch.compile(lambda bs, rs, bi, re, ro: torch_d2d_56l(bs, rs, bi, re, ro))
            with torch.cuda.stream(stream):
                c_d2d(buf_dev, raw_layers, reqs[0][3], reqs[0][4], reqs[0][5])
            torch.cuda.synchronize()
            compile_d2d_us = measure_stream(
                lambda: c_d2d(buf_dev, raw_layers, reqs[0][3], reqs[0][4], reqs[0][5]),
                stream,
                reps=reps,
            )
            # Manual CUDA graph with static GPU index slots
            b_st, re_st, ro_st = reqs[0][3].clone(), reqs[0][4].clone(), reqs[0][5].clone()
            with torch.cuda.stream(stream):
                for _ in range(3):
                    c_d2d(buf_dev, raw_layers, b_st, re_st, ro_st)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                c_d2d(buf_dev, raw_layers, b_st, re_st, ro_st)
            torch.cuda.synchronize()
            compile_cg_d2d_us = measure_stream(g.replay, stream, reps=reps)
        except Exception:
            pass

    # CopyEngine StepPlan (D2D GPU indices & H2D host indices)
    ce_d2d_us = ce_h2d_us = ce_einops_us = None
    ce_d2d_strat = ce_h2d_strat = None
    reblock_parity_ok = None
    if hasattr(ce, "plan"):
        eng_d = ce.Engine()
        def step_d2d(bs, rs, bi, re, ro):
            with eng_d.scope():
                for sb_t, rb_t in zip(bs, rs):
                    eng_d.copy_(ce.lazy(rb_t).index_select(0, re), ce.lazy(sb_t).slice(2, 0, 512).index_select(0, bi).permute(0, 2, 1, 3, 4))
                    eng_d.copy_(ce.lazy(rb_t).index_select(0, ro), ce.lazy(sb_t).slice(2, 512, 1024).index_select(0, bi).permute(0, 2, 1, 3, 4))
        p_d = eng_d.plan(step_d2d)
        with torch.cuda.stream(stream):
            for k in range(75):
                r = reqs[k % len(reqs)]
                p_d(buf_dev, raw_layers, r[3], r[4], r[5])
        torch.cuda.synchronize()

        # Bit-exact check on unseen reqs[27]
        with torch.cuda.stream(stream):
            for t in raw_layers:
                t.zero_()
            torch_d2d_56l(buf_dev, raw_layers, reqs[27][3], reqs[27][4], reqs[27][5])
        torch.cuda.synchronize()
        refs = [t.clone() for t in raw_layers]
        with torch.cuda.stream(stream):
            for t in raw_layers:
                t.zero_()
            p_d(buf_dev, raw_layers, reqs[27][3], reqs[27][4], reqs[27][5])
        torch.cuda.synchronize()
        reblock_parity_ok = all(torch.equal(a, b) for a, b in zip(raw_layers, refs))

        def run_ce_d2d():
            k = it[0] % len(reqs); it[0] += 1
            r = reqs[k]
            p_d(buf_dev, raw_layers, r[3], r[4], r[5])
        ce_d2d_us = measure_stream(run_ce_d2d, stream, reps=reps)
        ce_d2d_strat = eng_d.stats()["last"]["decisions"][0]["strategy"]

        eng_h = ce.Engine()
        def step_h2d(bs, rs, bi, re, ro):
            with eng_h.scope():
                for sb_t, rb_t in zip(bs, rs):
                    eng_h.copy_(ce.lazy(rb_t).index_select(0, re), ce.lazy(sb_t).slice(2, 0, 512).index_select(0, bi).permute(0, 2, 1, 3, 4))
                    eng_h.copy_(ce.lazy(rb_t).index_select(0, ro), ce.lazy(sb_t).slice(2, 512, 1024).index_select(0, bi).permute(0, 2, 1, 3, 4))
        p_h = eng_h.plan(step_h2d)
        with torch.cuda.stream(stream):
            for k in range(75):
                r = reqs[k % len(reqs)]
                p_h(buf_pin, raw_layers, r[0], r[1], r[2])
        torch.cuda.synchronize()

        def run_ce_h2d():
            k = it[0] % len(reqs); it[0] += 1
            r = reqs[k]
            p_h(buf_pin, raw_layers, r[0], r[1], r[2])
        ce_h2d_us = measure_stream(run_ce_h2d, stream, reps=reps)
        ce_h2d_strat = eng_h.stats()["last"]["decisions"][0]["strategy"]

        # Test einops backend composition if supported (v0.5+)
        if einops is not None:
            try:
                eng_ei = ce.Engine()
                def step_ei(bs, rs, bi, re, ro):
                    with eng_ei.scope():
                        for sb_t, rb_t in zip(bs, rs):
                            vs = einops.rearrange(ce.lazy(sb_t), "b h (half s) kv d -> half b s h kv d", half=2, s=512)
                            eng_ei.copy_(ce.lazy(rb_t).index_select(0, re), vs.select(0, 0).index_select(0, bi))
                            eng_ei.copy_(ce.lazy(rb_t).index_select(0, ro), vs.select(0, 1).index_select(0, bi))
                p_ei = eng_ei.plan(step_ei)
                with torch.cuda.stream(stream):
                    for k in range(75):
                        r = reqs[k % len(reqs)]
                        p_ei(buf_dev, raw_layers, r[3], r[4], r[5])
                torch.cuda.synchronize()
                ce_einops_us = measure_stream(lambda: p_ei(buf_dev, raw_layers, reqs[0][3], reqs[0][4], reqs[0][5]), stream, reps=reps)
            except Exception:
                ce_einops_us = None

    # Case 4B: Unaligned prefix-hit segments (`seq_begin = 137` -> exposes eager Plan-Miss overhead)
    sb_u, se_u = 137, 137 + 8192
    runs_u = []
    b_ids_u, r_ids_u = reqs[0][0].tolist() + [10, 11], reqs[0][1].tolist() + reqs[0][2].tolist() + [30, 31, 32, 33]
    for sbi in range(sb_u // S_buf, (se_u + S_buf - 1) // S_buf):
        gb, ge = max(sb_u, sbi * S_buf), min(se_u, (sbi + 1) * S_buf)
        for dbi in range(gb // S_raw, (ge + S_raw - 1) // S_raw):
            lo, hi = max(gb, dbi * S_raw), min(ge, (dbi + 1) * S_raw)
            runs_u.append((b_ids_u[sbi], lo - sbi * S_buf, r_ids_u[dbi], lo - dbi * S_raw, hi - lo))

    eng_u = ce.Engine()
    def ce_unaligned_scope():
        with eng_u.scope():
            for sb_t, rb_t in zip(buf_dev, raw_layers):
                for sbid, soff, dbid, doff, ln in runs_u:
                    rb_t[dbid, doff : doff + ln].copy_(sb_t[sbid, :, soff : soff + ln].transpose(0, 1), non_blocking=True)
    with torch.cuda.stream(stream):
        for _ in range(70):
            ce_unaligned_scope()
    torch.cuda.synchronize()
    ce_unaligned_frozen_us = measure_stream(eng_u.freeze().run, stream, reps=reps)

    del raw_layers, buf_dev, buf_pin, stage_h2d
    cleanup_cuda()
    return {
        "payload_mb": round(8192 * sum(heads) * 2 * D / 1e6, 2),
        "d2d_eager_us": eager_d2d_us,
        "d2d_compile_us": compile_d2d_us,
        "d2d_compile_cudagraph_us": compile_cg_d2d_us,
        "d2d_ce_step_plan_us": ce_d2d_us,
        "d2d_ce_strategy": ce_d2d_strat,
        "d2d_parity_ok": reblock_parity_ok if hasattr(ce, "plan") else None,
        "d2d_ce_einops_step_plan_us": ce_einops_us,
        "d2d_unaligned_frozen_us": ce_unaligned_frozen_us,
        "h2d_eager_us": eager_h2d_us,
        "h2d_ce_step_plan_us": ce_h2d_us,
        "h2d_ce_strategy": ce_h2d_strat,
    }


# ---------------------------------------------------------------------------
# Suite 5: TP2 -> TP4 Resharding (24 layers, 96 tensor pairs, 1536 head slices)
# ---------------------------------------------------------------------------

def suite_reshard(stream: torch.cuda.Stream, reps: int, quick: bool, **_) -> Dict[str, Any]:
    # Each TP2 rank holds h heads of a layer ([2 (K/V), 32 pages, h, S, D]); TP4 rank r takes heads
    # (r % 2) * h/2 .. (r % 2 + 1) * h/2 of TP2 rank r // 2, for nb page pairs.
    S, D, nb, P = 512, 128, 8, 32
    heads = HETEROGENEOUS_24L_HEADS  # per TP2 rank, all even
    src_dev = [[torch.randint(1, 120, (2, P, h, S, D), dtype=torch.uint8, device=DEV) for h in heads] for _ in range(2)]
    src_pin = [[t.cpu().pin_memory() for t in rank] for rank in src_dev]
    dst_shards = [[torch.zeros((2, P, h // 2, S, D), dtype=torch.uint8, device=DEV) for h in heads] for _ in range(4)]

    rng = np.random.default_rng(123)
    src_pages = rng.permutation(P)[:nb].tolist()
    dst_pages = rng.permutation(P)[:nb].tolist()

    def build_slice_list(srcs):
        ops, total_b = [], 0
        for r_dst in range(4):
            for l_idx in range(len(heads)):
                st, dt = srcs[r_dst // 2][l_idx], dst_shards[r_dst][l_idx]
                h_dst = dt.shape[2]
                h0 = (r_dst % 2) * h_dst
                for side in range(2):
                    for sp, dp in zip(src_pages, dst_pages):
                        dv, sv = dt[side, dp], st[side, sp, h0 : h0 + h_dst]
                        assert dv.shape == sv.shape
                        ops.append((dv, sv))
                        total_b += dv.numel()
        return ops, total_b

    def build_pair_segments(srcs):
        segs = []
        for r_dst in range(4):
            for l_idx in range(len(heads)):
                st, dt = srcs[r_dst // 2][l_idx], dst_shards[r_dst][l_idx]
                h_src, h_dst = st.shape[2], dt.shape[2]
                h0 = (r_dst % 2) * h_dst
                so, do = [], []
                for side in range(2):
                    for sp, dp in zip(src_pages, dst_pages):
                        so.append(((side * P + sp) * h_src + h0) * S * D)
                        do.append((side * P + dp) * h_dst * S * D)
                ln = np.full(len(so), h_dst * S * D, np.int64)
                segs.append((st.view(-1), dt.view(-1), np.asarray(so, np.int64), np.asarray(do, np.int64), ln))
        return segs

    def zero_dst():
        with torch.cuda.stream(stream):
            for rank in dst_shards:
                for t in rank:
                    t.zero_()

    def snapshot():
        torch.cuda.synchronize()
        return [t.clone() for rank in dst_shards for t in rank]

    def same(ref):
        torch.cuda.synchronize()
        return all(torch.equal(a, b) for a, b in zip([t for rank in dst_shards for t in rank], ref))

    out: Dict[str, Any] = {}
    for tag, srcs in (("d2d", src_dev), ("h2d", src_pin)):
        ops, total_bytes = build_slice_list(srcs)
        segs = build_pair_segments(srcs)
        dsts, srcl = [d for d, _ in ops], [x for _, x in ops]

        def torch_loop():
            for dv, sv in ops:
                dv.copy_(sv, non_blocking=True)

        zero_dst()
        with torch.cuda.stream(stream):
            torch_loop()
        ref = snapshot()

        r: Dict[str, Any] = {}
        r["torch_loop_us"] = measure_stream(torch_loop, stream, warmup=2, reps=max(5, reps // 2))
        r["torch_foreach_us"] = measure_stream(
            lambda: torch._foreach_copy_(dsts, srcl, non_blocking=True), stream, warmup=2, reps=reps)
        try:
            g = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=stream):
                torch_loop()
            zero_dst()
            with torch.cuda.stream(stream):
                g.replay()
            r["torch_loop_cudagraph_ok"] = same(ref)
            r["torch_loop_cudagraph_us"] = measure_stream(g.replay, stream, reps=reps)
        except Exception as e:
            r["torch_loop_cudagraph_us"] = short_err(e)

        eng = ce.Engine()

        def fn():
            with eng.scope():
                for sf, df, so, do, ln in segs:
                    eng.segments(sf, df, so, do, ln)

        with torch.cuda.stream(stream):
            for _ in range(75):
                fn()
        zero_dst()
        with torch.cuda.stream(stream):
            fn()
        r["ce_parity_ok"] = same(ref)
        frz = eng.freeze()
        r["ce_scope_us"] = measure_stream(fn, stream, warmup=2, reps=max(5, reps // 2))
        r["ce_frozen_us"] = measure_stream(frz.run, stream, reps=reps)
        r["ce_strategy"] = eng.stats()["last"]["decisions"][0]["strategy"]
        out[tag] = r
        out["slice_ops"], out["tensor_pairs"], out["payload_mb"] = len(ops), len(segs), round(total_bytes / 1e6, 2)

    del src_dev, src_pin, dst_shards
    cleanup_cuda()
    # Flat keys kept for --compare against older reports.
    out["d2d_ce_frozen_us"] = out["d2d"]["ce_frozen_us"]
    return out


# ---------------------------------------------------------------------------
# Suite 6: Pure D2D & Narrow-Row Transpose (`bulk` TMA & `tma2d`) Across SM Caps
# ---------------------------------------------------------------------------

def _tma2d_subprocess_probe(sms: int, reps: int) -> Dict[str, Any]:
    """Run tma2d narrow-row transpose in an isolated subprocess so any driver/PTX mismatch cannot poison main CUDA ctx."""
    code = f"""
import json, statistics, time, torch, copyengine as ce
DEV = "cuda:0"
stream = torch.cuda.Stream()
s = torch.randn((64, 512, 16, 128), dtype=torch.bfloat16, device=DEV)
d = torch.empty((64, 16, 512, 128), dtype=torch.bfloat16, device=DEV)
e = ce.Engine(max_sms={sms}, tune_topk=6)
for _ in range(90):
    with torch.cuda.stream(stream):
        e.copy_(d, s.permute(0, 2, 1, 3))
torch.cuda.synchronize()
with torch.cuda.stream(stream):
    d.zero_()
    e.copy_(d, s.permute(0, 2, 1, 3))
torch.cuda.synchronize()
assert torch.equal(d, s.permute(0, 2, 1, 3)), "tma2d parity failed"
frz = e.freeze()
ts = []
for _ in range({reps}):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        e0.record(stream); frz.run(); e1.record(stream); stream.synchronize()
    ts.append(round(e0.elapsed_time(e1) * 1e3, 1))
dec = e.stats()["last"]["decisions"][0]["strategy"]
print("JSON:" + json.dumps({{"us": statistics.median(ts), "strategy": dec}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=90,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip().splitlines()[-1] if (proc.stderr or proc.stdout) else "crash"
        return {"error": err[:160]}
    for line in proc.stdout.splitlines():
        if line.startswith("JSON:"):
            return json.loads(line[5:])
    return {"error": "no output"}


def suite_tma_sm_cap(stream: torch.cuda.Stream, reps: int, quick: bool, **_) -> Dict[str, Any]:
    # 6A: 1D segmented D2D (896 x 160 KB = 146.8 MB) across SM caps
    src = torch.randint(1, 120, (300 * 1024 * 1024,), dtype=torch.uint8, device=DEV)
    dst = torch.zeros_like(src)
    n_chunks, seg_b = 896, 163840
    rng = np.random.default_rng(42)
    so = (rng.permutation(src.numel() // seg_b)[:n_chunks] * seg_b).astype(np.int64)
    do = (rng.permutation(src.numel() // seg_b)[:n_chunks] * seg_b).astype(np.int64)
    ln = np.full(n_chunks, seg_b, dtype=np.int64)
    mb_1d = round(n_chunks * seg_b / 1e6, 2)

    sm_caps = [0, 8] if quick else [0, 8, 4]
    d2d_1d = []
    for sms in sm_caps:
        eng = ce.Engine(max_sms=sms, tune_topk=5)
        with torch.cuda.stream(stream):
            for _ in range(95):
                eng.segments(src, dst, so, do, ln)
        torch.cuda.synchronize()
        us = measure_stream(eng.freeze().run, stream, reps=reps)[1]
        strat = eng.stats()["last"]["decisions"][0]["strategy"]
        d2d_1d.append({
            "max_sms": sms,
            "gpu_us": us,
            "gbps": round(mb_1d / us * 1e3, 1),
            "strategy": strat,
        })

    del src, dst
    cleanup_cuda()

    # 6B: Narrow-row 256B BF16 permute (`[64, 512, 16, 128] -> [64, 16, 512, 128]` = 134.22 MB)
    s2 = torch.randn((64, 512, 16, 128), dtype=torch.bfloat16, device=DEV)
    d2 = torch.empty((64, 16, 512, 128), dtype=torch.bfloat16, device=DEV)
    eager_2d = measure_stream(lambda: d2.copy_(s2.permute(0, 2, 1, 3), non_blocking=True), stream, reps=reps)[1]
    del s2, d2
    cleanup_cuda()

    narrow_2d = []
    tma2d_hw = cc() >= (9, 0)
    for sms in sm_caps:
        r = _tma2d_subprocess_probe(sms, reps)
        r["max_sms"] = sms
        r["tma2d_hw"] = tma2d_hw  # below sm90 the same copy runs on the row-wise strategies
        if "us" in r:
            r["gbps"] = round(134.22 / r["us"] * 1e3, 1)
        narrow_2d.append(r)

    return {
        "d2d_segmented_147mb": d2d_1d,
        "narrow_row_permute_134mb_eager_us": eager_2d,
        "narrow_row_permute_134mb_by_sm": narrow_2d,
    }


# ---------------------------------------------------------------------------
# Suite 7: Foreground Interference Under a Continuous Background Load
# ---------------------------------------------------------------------------

class _Gate:
    """Holds a stream back until the host has enqueued its work, then releases it at once."""

    def __init__(self) -> None:
        self.stream = torch.cuda.Stream()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record(self.stream)
        with torch.cuda.stream(self.stream):
            torch.cuda._sleep(20_000_000)
        e1.record(self.stream)
        torch.cuda.synchronize()
        self.cycles_per_us = 20_000_000 / max(1.0, e0.elapsed_time(e1) * 1e3)
        self.us = 20_000.0

    def close(self, st: torch.cuda.Stream) -> None:
        self.stream.wait_stream(st)  # after the warm-up run st has queued
        with torch.cuda.stream(self.stream):
            torch.cuda._sleep(int(self.us * self.cycles_per_us))
        st.wait_event(self.stream.record_event())


def _gated_solo(fg_fn, fg_stream, gate: _Gate, reps: int) -> float:
    """The foreground alone, released by the gate like under load (device time, no host gaps)."""
    with torch.cuda.stream(fg_stream):
        fg_fn()
    ts = []
    while len(ts) < reps:
        flush_l2()
        torch.cuda.synchronize()
        with torch.cuda.stream(fg_stream):
            fg_fn()  # warm-up: clocks up, as with the background running
        t0 = time.perf_counter()
        gate.close(fg_stream)
        fg0, fg1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        fg0.record(fg_stream)
        with torch.cuda.stream(fg_stream):
            fg_fn()
        fg1.record(fg_stream)
        host_us = (time.perf_counter() - t0) * 1e6
        torch.cuda.synchronize()
        if host_us > 0.8 * gate.us:
            gate.us = min(2 * gate.us, 1e6)
            continue
        ts.append(fg0.elapsed_time(fg1) * 1e3)
    return round(statistics.median(ts), 1)


def _overlap(fg_fn, fg_solo_us, bg_fn, bg_bytes, fg_stream, bg_stream, gate: _Gate, reps: int) -> Dict[str, Any]:
    """Foreground slowdown while bg_fn runs back to back (a thread keeps two iterations in flight)
    for the whole foreground window, and the bytes the background moves inside that window."""
    import threading

    with torch.cuda.stream(bg_stream):
        bg_fn()
    with torch.cuda.stream(fg_stream):
        fg_fn()
    torch.cuda.synchronize()
    fg_t, gbps, solo_t = [], [], []
    tries = 0
    while len(fg_t) < reps and tries < 3 * reps:
        tries += 1
        flush_l2()
        torch.cuda.synchronize()
        marks: List[torch.cuda.Event] = []
        stop = threading.Event()
        failure: List[BaseException] = []

        def loop():
            try:
                with torch.cuda.stream(bg_stream):
                    e = torch.cuda.Event(enable_timing=True)
                    e.record(bg_stream)
                    marks.append(e)
                    while not stop.is_set():
                        if len(marks) >= 3:
                            marks[-3].synchronize()  # at most two iterations queued ahead
                        bg_fn()
                        e = torch.cuda.Event(enable_timing=True)
                        e.record(bg_stream)
                        marks.append(e)
            except BaseException as ex:  # surfaced below
                failure.append(ex)

        th = threading.Thread(target=loop, daemon=True)
        th.start()
        while len(marks) < 3 and th.is_alive():  # the background is in steady state
            time.sleep(0.0005)
        with torch.cuda.stream(fg_stream):
            fg_fn()  # warm-up: clocks up, as with the background running
        t0 = time.perf_counter()
        gate.close(fg_stream)
        fg0, fg1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        fg0.record(fg_stream)
        with torch.cuda.stream(fg_stream):
            fg_fn()
        fg1.record(fg_stream)
        host_us = (time.perf_counter() - t0) * 1e6
        fg1.synchronize()
        stop.set()
        th.join()
        torch.cuda.synchronize()
        if failure:
            raise failure[0]
        if host_us > 0.8 * gate.us:  # the gate opened before the foreground was enqueued
            gate.us = min(2 * gate.us, 1e6)
            continue
        fg_us = fg0.elapsed_time(fg1) * 1e3
        t = [fg0.elapsed_time(e) * 1e3 for e in marks]  # iteration i spans (t[i], t[i+1]]
        if t[-1] < fg_us:  # the background stopped inside the window
            continue
        moved = sum(bg_bytes * max(0.0, min(b, fg_us) - max(a, 0.0)) / (b - a)
                    for a, b in zip(t, t[1:]) if b > a)
        fg_t.append(fg_us)
        gbps.append(moved / fg_us / 1e3)
        solo_t.append(_gated_solo(fg_fn, fg_stream, gate, 1))  # paired: drift and clocks cancel
    if not fg_t:
        return {"error": "the foreground window could not be covered"}
    fg_med, solo_med = statistics.median(fg_t), statistics.median(solo_t)
    return {
        "fg_us": round(fg_med, 1),
        "fg_solo_paired_us": round(solo_med, 1),
        "fg_slowdown_pct": round((fg_med / solo_med - 1.0) * 100.0, 2),
        "bg_gbps_in_window": round(statistics.median(gbps), 2),
    }


def suite_interference(stream: torch.cuda.Stream, reps: int, max_sms: int, with_compile: bool = True, **_) -> Dict[str, Any]:
    L, B_dev, B_host, S, H, D = 56, 64, 32, 512, 2, 128
    nb = 24  # 352 MB H2D per background iteration
    dt_fg = torch.bfloat16 if bf16_ok() else torch.float16
    dev_layers = [torch.zeros((2, B_dev, H, S, D), dtype=torch.uint8, device=DEV) for _ in range(L)]
    host_arena = torch.randint(1, 100, (B_host, L, S, H, 2, D), dtype=torch.uint8).pin_memory()
    stage = torch.empty((nb, S, H, 2, D), dtype=torch.uint8, device=DEV)
    bg_bytes = nb * L * S * H * 2 * D
    fg_stream, gate = torch.cuda.Stream(), _Gate()

    def sized(fn_once, target_us=10_000.0):
        one = measure_stream(fn_once, fg_stream, warmup=2, reps=3)[1]
        n = max(1, int(target_us / max(one, 1.0)))
        return (lambda: [fn_once() for _ in range(n)]), n

    A_pre = torch.randn(512, 3584, dtype=dt_fg, device=DEV)
    B_pre = torch.randn(3584, 7168, dtype=dt_fg, device=DEV)
    C_pre = torch.empty(512, 7168, dtype=dt_fg, device=DEV)
    A_dec = torch.randn(16, 4096, dtype=dt_fg, device=DEV)
    B_dec = torch.randn(4096, 14336, dtype=dt_fg, device=DEV)
    C_dec = torch.empty(16, 14336, dtype=dt_fg, device=DEV)
    R_dec = torch.randn(16, 14336, dtype=dt_fg, device=DEV)

    def dec_once():
        torch.matmul(A_dec, B_dec, out=C_dec)
        C_dec.add_(R_dec)

    fg_prefill, n_pre = sized(lambda: torch.matmul(A_pre, B_pre, out=C_pre))  # compute bound
    fg_decode, n_dec = sized(dec_once)  # memory bound
    fg_pre_solo = _gated_solo(fg_prefill, fg_stream, gate, reps)
    fg_dec_solo = _gated_solo(fg_decode, fg_stream, gate, reps)

    # Paged: random host pages -> random device blocks.
    rng = np.random.default_rng(77)
    h_idx = rng.permutation(B_host)[:nb].astype(np.int64)
    d_idx = rng.permutation(B_dev)[:nb].astype(np.int64)
    d_idx_gpu = torch.from_numpy(d_idx).to(DEV)
    h_ids = h_idx.tolist()

    def bg_eager_h2d():
        for l_idx, dt in enumerate(dev_layers):
            torch._foreach_copy_([stage[j] for j in range(nb)], [host_arena[i, l_idx] for i in h_ids], non_blocking=True)
            dt.index_copy_(1, d_idx_gpu, stage.permute(3, 0, 2, 1, 4))

    def run_modes(runners, bytes_per_iter):
        rows = []
        for label, fn, strat in runners:
            try:
                bg_solo = measure_stream(fn, stream, warmup=2, reps=max(3, reps // 3))[1]
                rows.append({
                    "mode": label,
                    "strategy": strat,
                    "bg_solo_us": bg_solo,
                    "bg_solo_gbps": round(bytes_per_iter / bg_solo / 1e3, 2),
                    "prefill": _overlap(fg_prefill, fg_pre_solo, fn, bytes_per_iter, fg_stream, stream, gate, reps),
                    "decode": _overlap(fg_decode, fg_dec_solo, fn, bytes_per_iter, fg_stream, stream, gate, reps),
                })
            except Exception as e:
                rows.append({"mode": label, "error": short_err(e)})
        return rows

    runners_h2d = [("PyTorch eager", bg_eager_h2d, "eager")]
    if with_compile and dynamo is not None:
        try:
            reset_dynamo()
            c_h2d = torch.compile(bg_eager_h2d)
            with torch.cuda.stream(stream):
                c_h2d()
            torch.cuda.synchronize()
            runners_h2d.append(("torch.compile", c_h2d, "compile"))
        except Exception:
            pass

    settings = [("ce uncapped", 0, None)] + [(f"ce max_sms={m}", m, None) for m in sorted({16, max_sms, 4}, reverse=True)] + [
        (f"ce bandwidth={int(b / 1e9)}GBps", 0, b) for b in (32e9, 16e9, 8e9)]
    for label, sms, bw in settings:
        eng = ce.Engine(max_sms=sms, bandwidth=bw)

        def bg_fn(e=eng):
            with e.scope():
                for l_idx, dt in enumerate(dev_layers):
                    src_v = ce.lazy(host_arena[:, l_idx]).index_select(0, h_idx).permute(3, 0, 2, 1, 4)
                    e.copy_(ce.lazy(dt).index_select(1, d_idx), src_v)

        with torch.cuda.stream(stream):
            for _ in range(60):
                bg_fn()
        torch.cuda.synchronize()
        frz = eng.freeze()
        strat = eng.stats()["last"]["decisions"][0]["strategy"]
        runners_h2d.append((label, lambda f=frz: f.run(stream.cuda_stream), strat))

    h2d_modes = run_modes(runners_h2d, bg_bytes)
    del dev_layers, host_arena, stage
    cleanup_cuda()

    # Scenario B: SM-heavy background (24-layer paged INT4 quantization on the GPU)
    sm_heavy_modes: List[Dict[str, Any]] = []
    cq = get_quant_ref()
    if hasattr(ce, "quantize") and cq is not None and bf16_ok():
        x24 = [torch.randn((48, 512, 2 * h, 128), dtype=torch.bfloat16, device=DEV) for h in HETEROGENEOUS_24L_HEADS]
        q24 = [torch.zeros((48, 512, 2 * h, 64), dtype=torch.uint8, device=DEV) for h in HETEROGENEOUS_24L_HEADS]
        s24 = [torch.zeros((48, 512, 2 * h, 1), dtype=torch.float32, device=DEV) for h in HETEROGENEOUS_24L_HEADS]
        sp_gpu = torch.randperm(48, device=DEV)[:16]
        ds_gpu = torch.randperm(48, device=DEV)[:16]
        q_bytes = sum(16 * x[0].nbytes for x in x24)

        def bg_q_eager():
            for x, q, sc in zip(x24, q24, s24):
                cq.native_quantize(x, q, sc, None, "int4", 128, 1e-12, 1.0, sp_gpu, ds_gpu)

        q_runners = [("PyTorch eager", bg_q_eager, "eager")]
        if with_compile and dynamo is not None:
            try:
                reset_dynamo()
                c_q = torch.compile(bg_q_eager)
                with torch.cuda.stream(stream):
                    c_q()
                torch.cuda.synchronize()
                q_runners.append(("torch.compile", c_q, "compile"))
            except Exception:
                pass
        for m in [0] + sorted({16, max_sms, 4}, reverse=True):
            eq = ce.Engine(max_sms=m)

            def step_q(xs, qs, ss, sp_, ds_, e=eq):
                with e.scope():
                    for x, q, sc in zip(xs, qs, ss):
                        e.quantize(x, q, sc, fmt="int4", group=128, x_index=sp_, index=ds_)

            pl = eq.plan(step_q)
            with torch.cuda.stream(stream):
                for _ in range(75):
                    pl(x24, q24, s24, sp_gpu, ds_gpu)
            torch.cuda.synchronize()
            q_runners.append(("ce uncapped" if m == 0 else f"ce max_sms={m}", lambda p=pl: p(x24, q24, s24, sp_gpu, ds_gpu), "tile"))
        sm_heavy_modes = run_modes(q_runners, q_bytes)
        del x24, q24, s24, sp_gpu, ds_gpu

    del A_pre, B_pre, C_pre, A_dec, B_dec, C_dec, R_dec
    cleanup_cuda()
    return {
        "fg_dtype": str(dt_fg),
        "fg_prefill_gemms": n_pre,
        "fg_prefill_solo_us": fg_pre_solo,
        "fg_decode_gemvs": n_dec,
        "fg_decode_solo_us": fg_dec_solo,
        "h2d_bg_mb": round(bg_bytes / 1e6, 1),
        "h2d_modes": h2d_modes,
        "sm_heavy_quant_modes": sm_heavy_modes,
    }


# ---------------------------------------------------------------------------
# Suite 8: 24-Layer Tile IR GPU Paged Gather + Quantization & Transposed V-Cache
# ---------------------------------------------------------------------------

def suite_tile_quant(
    stream: torch.cuda.Stream,
    reps: int,
    quick: bool,
    with_compile: bool,
    max_sms: int,
    **_,
) -> Dict[str, Any]:
    if not hasattr(ce, "quantize"):
        return {"status": "skipped", "reason": "copyengine.quantize unavailable in this version"}
    cq = get_quant_ref()
    if cq is None:
        return {"status": "skipped", "reason": "copyengine._quant reference module unavailable"}

    num_pages, S, D, nb = 48, 512, 128, 8
    heads = HETEROGENEOUS_24L_HEADS
    bf16_layers = [torch.randn((num_pages, S, 2 * h, D), dtype=torch.bfloat16, device=DEV) for h in heads]
    out_layers = [torch.empty_like(x) for x in bf16_layers]
    raw_mb = round(sum(nb * t[0].nbytes for t in bf16_layers) / 1e6, 2)

    rng = np.random.default_rng(2026)
    sp0 = torch.tensor(rng.permutation(num_pages)[:nb], dtype=torch.int64, device=DEV)
    ds0 = torch.tensor(rng.permutation(num_pages)[:nb], dtype=torch.int64, device=DEV)
    sp1 = torch.tensor(rng.permutation(num_pages)[:nb], dtype=torch.int64, device=DEV)
    ds1 = torch.tensor(rng.permutation(num_pages)[:nb], dtype=torch.int64, device=DEV)

    fmts = [("fp8_e4m3", 128, False), ("int4", 128, False), ("nvfp4", 16, False)] if quick else [
        ("fp8_e4m3", 128, False),
        ("fp8_e4m3", 128, True),
        ("mxfp8_e4m3", 32, False),
        ("int8", 128, False),
        ("int4", 128, False),
        ("uint4", 128, False),
        ("nvfp4", 16, False),
    ]

    rows = []
    for fmt, grp, is_static in fmts:
        if fmt.startswith("mx") and not has_dtype("float8_e8m0fnu"):
            rows.append({"format": fmt, "status": "skipped", "reason": "torch has no float8_e8m0fnu"})
            continue
        reset_dynamo()
        packed = fmt in ("int4", "uint4", "nvfp4")
        q_dt = torch.float8_e4m3fn if fmt in ("fp8_e4m3", "mxfp8_e4m3") else (torch.int8 if fmt == "int8" else torch.uint8)
        s_dt = torch.float8_e8m0fnu if fmt.startswith("mx") else (torch.float8_e4m3fn if fmt == "nvfp4" else torch.float32)
        gs = 0.25 if fmt == "nvfp4" else 1.0

        q_layers = [torch.zeros((num_pages, S, 2 * h, D // 2 if packed else D), dtype=q_dt, device=DEV) for h in heads]
        s_layers = [(torch.ones((num_pages, S, 2 * h, D // grp), dtype=torch.float32, device=DEV) * 0.05).to(s_dt) for h in heads]
        z_layers = [torch.zeros((num_pages, S, 2 * h, D // grp), dtype=torch.uint8, device=DEV) if fmt == "uint4" else None for h in heads]
        z_args = [z for z in z_layers if z is not None]

        def torch_q_24l(sp, ds):
            for x, q, s, z in zip(bf16_layers, q_layers, s_layers, z_layers):
                if is_static:
                    cq.native_quantize_static(x, q, s, z, fmt, grp, gs, sp, ds)
                else:
                    cq.native_quantize(x, q, s, z, fmt, grp, 1e-12, gs, sp, ds)

        def torch_dq_24l(sp, ds):
            for o, q, s, z in zip(out_layers, q_layers, s_layers, z_layers):
                cq.native_dequantize(q, s, z, o, fmt, grp, gs, sp, ds)

        eng = ce.Engine()
        eng_sm = ce.Engine(max_sms=max_sms)
        def make_q_step(e):
            def step_q(xs, qs, ss, zs, sp, ds):
                with e.scope():
                    for i in range(len(xs)):
                        e.quantize(
                            xs[i], qs[i], ss[i], zs[i] if zs else None,
                            fmt=fmt, group=grp, global_scale=gs, static=is_static, x_index=sp, index=ds,
                        )
            return step_q

        plan_q = eng.plan(make_q_step(eng))
        plan_q_sm = eng_sm.plan(make_q_step(eng_sm))
        with torch.cuda.stream(stream):
            for i in range(75):
                sp, ds = (sp0, ds0) if i % 2 == 0 else (sp1, ds1)
                plan_q(bf16_layers, q_layers, s_layers, z_args, sp, ds)
                plan_q_sm(bf16_layers, q_layers, s_layers, z_args, sp, ds)
        torch.cuda.synchronize()

        # Bit-exact parity against reference recipe on unseen (sp1, ds0)
        with torch.cuda.stream(stream):
            for q in q_layers:
                q.view(torch.uint8).zero_()
            if not is_static:
                for s in s_layers:
                    s.view(torch.uint8).zero_()
            for z in z_args:
                z.zero_()
            torch_q_24l(sp1, ds0)
            ref_q = [q.view(torch.uint8).clone() for q in q_layers]
            ref_s = [s.view(torch.uint8).clone() for s in s_layers]
            ref_z = [z.clone() for z in z_args]
        torch.cuda.synchronize()

        with torch.cuda.stream(stream):
            for q in q_layers:
                q.view(torch.uint8).zero_()
            if not is_static:
                for s in s_layers:
                    s.view(torch.uint8).zero_()
            for z in z_args:
                z.zero_()
            plan_q(bf16_layers, q_layers, s_layers, z_args, sp1, ds0)
        torch.cuda.synchronize()
        for i in range(len(q_layers)):
            assert torch.equal(q_layers[i].view(torch.uint8), ref_q[i]), f"{fmt} q mismatch L{i}"
            assert torch.equal(s_layers[i].view(torch.uint8), ref_s[i]), f"{fmt} scale mismatch L{i}"
        for i in range(len(z_args)):
            assert torch.equal(z_args[i], ref_z[i]), f"{fmt} zero mismatch L{i}"

        eager_q_us = measure_stream(lambda: torch_q_24l(sp0, ds0), stream, reps=reps)[1]
        ce_q_us = measure_stream(lambda: plan_q(bf16_layers, q_layers, s_layers, z_args, sp0, ds0), stream, reps=reps)[1]
        ce_q_sm_us = measure_stream(lambda: plan_q_sm(bf16_layers, q_layers, s_layers, z_args, sp0, ds0), stream, reps=reps)[1]

        comp_q_us = None
        comp_q_bit_diffs = None
        comp_q_err = None
        if with_compile and dynamo is not None and not is_static and fmt in ("fp8_e4m3", "nvfp4") and cc() < (8, 9):
            comp_q_err = "skipped: Triton fp8 needs sm89+"
        elif with_compile and dynamo is not None and not is_static and fmt in ("fp8_e4m3", "int4", "nvfp4"):
            try:
                c_q = torch.compile(lambda sp_, ds_: torch_q_24l(sp_, ds_))
                with torch.cuda.stream(stream):
                    c_q(sp1, ds0)
                torch.cuda.synchronize()
                # Count any 1-ULP differences due to Inductor fast reciprocal
                with torch.cuda.stream(stream):
                    for q in q_layers:
                        q.view(torch.uint8).zero_()
                    c_q(sp1, ds0)
                torch.cuda.synchronize()
                comp_q_bit_diffs = sum(int((q.view(torch.uint8) != rq).sum().item()) for q, rq in zip(q_layers, ref_q))
                comp_q_us = measure_stream(lambda: c_q(sp0, ds0), stream, reps=reps)[1]
            except Exception as e:
                comp_q_err = short_err(e)

        ce_dq_us = eager_dq_us = None
        if not is_static:
            eng_dq = ce.Engine()
            def step_dq(outs, qs, ss, zs, sp, ds):
                with eng_dq.scope():
                    for i in range(len(outs)):
                        eng_dq.dequantize(qs[i], ss[i], outs[i], zs[i] if zs else None, fmt=fmt, group=grp, global_scale=gs, index=ds, x_index=sp)
            plan_dq = eng_dq.plan(step_dq)
            with torch.cuda.stream(stream):
                for i in range(75):
                    sp, ds = (sp0, ds0) if i % 2 == 0 else (sp1, ds1)
                    plan_dq(out_layers, q_layers, s_layers, z_args, sp, ds)
            torch.cuda.synchronize()
            eager_dq_us = measure_stream(lambda: torch_dq_24l(sp0, ds0), stream, reps=reps)[1]
            ce_dq_us = measure_stream(lambda: plan_dq(out_layers, q_layers, s_layers, z_args, sp0, ds0), stream, reps=reps)[1]

        rows.append({
            "format": f"{fmt}_static" if is_static else fmt,
            "group": grp,
            "eager_quant_us": eager_q_us,
            "compile_quant_us": comp_q_us,
            "compile_bit_diffs_vs_recipe": comp_q_bit_diffs,
            "compile_error": comp_q_err,
            "ce_quant_us": ce_q_us,
            "ce_quant_sm_cap_us": ce_q_sm_us,
            "ce_calls": eng.stats()["last"]["calls"],
            "eager_dequant_us": eager_dq_us,
            "ce_dequant_us": ce_dq_us,
        })
        del q_layers, s_layers, z_layers
        cleanup_cuda()

    # Also test transposed V-cache quantization along token dimension (`[8, 512, 8, 128] -> transpose(1, 3)`, g=16 NVFP4)
    v_cache = torch.randn((8, 512, 8, 128), dtype=torch.bfloat16, device=DEV)
    v_t = v_cache.transpose(1, 3)
    vq = torch.zeros((8, 128, 8, 256), dtype=torch.uint8, device=DEV)
    vs = torch.zeros((8, 128, 8, 32), dtype=torch.float8_e4m3fn, device=DEV)
    cq.native_quantize(v_t, vq, vs, None, "nvfp4", 16, 1e-12, 0.25)
    ref_vq, ref_vs = vq.clone(), vs.view(torch.uint8).clone()
    vq.zero_(); vs.view(torch.uint8).zero_()
    eng_vt = ce.Engine()
    with torch.cuda.stream(stream):
        for _ in range(60):
            eng_vt.quantize(v_t, vq, vs, fmt="nvfp4", global_scale=0.25)
    torch.cuda.synchronize()
    assert torch.equal(vq, ref_vq) and torch.equal(vs.view(torch.uint8), ref_vs), "Transposed V-Cache NVFP4 mismatch!"
    vt_us = measure_stream(eng_vt.freeze().run, stream, reps=reps)[1]

    del bf16_layers, out_layers, v_cache, vq, vs
    cleanup_cuda()
    return {
        "raw_bf16_mb": raw_mb,
        "formats": rows,
        "transposed_vcache_nvfp4_67mb_us": vt_us,
    }


# ---------------------------------------------------------------------------
# Suite 9: Cross-PCIe Quantizing Offload & Onload (+ Paged Host Pool in v0.5)
# ---------------------------------------------------------------------------

def suite_pcie_quant(stream: torch.cuda.Stream, reps: int, quick: bool, with_compile: bool, **_) -> Dict[str, Any]:
    if not hasattr(ce, "quantize"):
        return {"status": "skipped", "reason": "copyengine.quantize unavailable in this version"}
    cq = get_quant_ref()
    if cq is None:
        return {"status": "skipped", "reason": "copyengine._quant reference unavailable"}

    nb = 8
    S, D = 512, 128
    heads = HETEROGENEOUS_24L_HEADS
    gpu_24l = [torch.randn((nb, S, 2 * h, D), dtype=torch.bfloat16, device=DEV) for h in heads]
    out_24l = [torch.empty_like(x) for x in gpu_24l]
    raw_mb = round(sum(t.nbytes for t in gpu_24l) / 1e6, 2)

    fmts = [("fp8_e4m3", 128), ("int4", 128), ("nvfp4", 16)] if quick else [
        ("fp8_e4m3", 128), ("mxfp8_e4m3", 32), ("int4", 128), ("nvfp4", 16)
    ]

    rows = []
    for fmt, grp in fmts:
        if fmt.startswith("mx") and not has_dtype("float8_e8m0fnu"):
            rows.append({"format": fmt, "status": "skipped", "reason": "torch has no float8_e8m0fnu"})
            continue
        reset_dynamo()
        packed = fmt in ("int4", "nvfp4")
        q_dt = torch.float8_e4m3fn if fmt in ("fp8_e4m3", "mxfp8_e4m3") else torch.uint8
        s_dt = torch.float8_e8m0fnu if fmt.startswith("mx") else (torch.float8_e4m3fn if fmt == "nvfp4" else torch.float32)
        gs = 0.25 if fmt == "nvfp4" else 1.0

        hq_24l = [torch.empty((nb, S, 2 * h, D // 2 if packed else D), dtype=q_dt, device="cpu").pin_memory() for h in heads]
        hs_24l = [torch.empty((nb, S, 2 * h, D // grp), dtype=s_dt, device="cpu").pin_memory() for h in heads]
        comp_mb = round(sum(q.nbytes + s.nbytes for q, s in zip(hq_24l, hs_24l)) / 1e6, 2)

        # Single-tensor equivalent for clean Inductor comparison
        x_1t = torch.randn((nb, S, 120, D), dtype=torch.bfloat16, device=DEV)
        hq_1t = torch.empty((nb, S, 120, D // 2 if packed else D), dtype=q_dt, device="cpu").pin_memory()
        hs_1t = torch.empty((nb, S, 120, D // grp), dtype=s_dt, device="cpu").pin_memory()
        out_1t = torch.empty_like(x_1t)

        comp_1t_d2h = comp_1t_h2d = comp_1t_quant_only = plain_d2h = None
        comp_err = None
        comp_suspect = None
        if with_compile and dynamo is not None and fmt in ("fp8_e4m3", "nvfp4") and cc() < (8, 9):
            comp_err = "skipped: Triton fp8 needs sm89+"
        elif with_compile and dynamo is not None and fmt in ("fp8_e4m3", "int4", "nvfp4"):
            try:
                def off_1t(x_, hq_, hs_):
                    codes, sc, bits = cq.quantize(x_, grp, fmt, 1e-12, gs)
                    hq_.view(torch.uint8).copy_(codes.view(hq_.shape), non_blocking=True)
                    if sc is not None:
                        hs_.copy_(sc.view(hs_.shape), non_blocking=True)
                    else:
                        hs_.view(torch.uint8).copy_(bits.view(hs_.shape), non_blocking=True)

                def on_1t(hq_, hs_, out_):
                    qg = hq_.view(torch.uint8).to(DEV, non_blocking=True).reshape(-1)
                    sg = hs_.to(DEV, non_blocking=True).reshape(-1) if hs_.dtype == torch.float32 else None
                    bg = None if sg is not None else hs_.view(torch.uint8).to(DEV, non_blocking=True).reshape(-1)
                    out_.copy_(cq.dequantize(qg, sg, bg, grp, fmt, gs).view(out_.shape).to(out_.dtype), non_blocking=True)

                c_off = torch.compile(off_1t)
                c_on = torch.compile(on_1t)
                with torch.cuda.stream(stream):
                    c_off(x_1t, hq_1t, hs_1t)
                    c_on(hq_1t, hs_1t, out_1t)
                torch.cuda.synchronize()
                comp_1t_d2h = measure_stream(lambda: c_off(x_1t, hq_1t, hs_1t), stream, reps=reps)[1]
                comp_1t_h2d = measure_stream(lambda: c_on(hq_1t, hs_1t, out_1t), stream, reps=reps)[1]
                # Sanity check of the D2H number: the same quantization compiled without the copy to the
                # host, plus a plain D2H of the outputs. Far above their sum means Inductor moved work to
                # the CPU (converting after the copy), a compile artifact rather than a GPU cost.
                c_qo = torch.compile(lambda x_: cq.quantize(x_, grp, fmt, 1e-12, gs))
                with torch.cuda.stream(stream):
                    c_qo(x_1t)
                torch.cuda.synchronize()
                comp_1t_quant_only = measure_stream(lambda: c_qo(x_1t), stream, reps=reps)[1]
                qd = torch.empty(hq_1t.shape, dtype=torch.uint8, device=DEV)
                sd = torch.empty(hs_1t.shape, dtype=hs_1t.dtype, device=DEV)

                def plain():
                    hq_1t.view(torch.uint8).copy_(qd, non_blocking=True)
                    hs_1t.copy_(sd, non_blocking=True)

                plain_d2h = measure_stream(plain, stream, reps=reps)[1]
                comp_suspect = comp_1t_d2h > 2.0 * (comp_1t_quant_only + plain_d2h)
                del qd, sd
            except Exception as e:
                comp_err = short_err(e)

        # CopyEngine 1-tensor
        eng_1d = ce.Engine()
        eng_1h = ce.Engine()
        with torch.cuda.stream(stream):
            for _ in range(75):
                eng_1d.quantize(x_1t, hq_1t, hs_1t, fmt=fmt, group=grp, global_scale=gs)
                eng_1h.dequantize(hq_1t, hs_1t, out_1t, fmt=fmt, group=grp, global_scale=gs)
        torch.cuda.synchronize()
        ce_1t_d2h = measure_stream(eng_1d.freeze().run, stream, reps=reps)[1]
        ce_1t_h2d = measure_stream(eng_1h.freeze().run, stream, reps=reps)[1]
        strat_1t_d2h = eng_1d.stats()["last"]["decisions"][0]["strategy"]
        strat_1t_h2d = eng_1h.stats()["last"]["decisions"][0]["strategy"]

        # CopyEngine 24-layer scope (tested safely for multi-op tile_staged)
        ce_24l_d2h = ce_24l_h2d = None
        try:
            eng_24d = ce.Engine()
            eng_24h = ce.Engine()
            with torch.cuda.stream(stream):
                for _ in range(75):
                    with eng_24d.scope():
                        for x, q, s in zip(gpu_24l, hq_24l, hs_24l):
                            eng_24d.quantize(x, q, s, fmt=fmt, group=grp, global_scale=gs)
                    with eng_24h.scope():
                        for o, q, s in zip(out_24l, hq_24l, hs_24l):
                            eng_24h.dequantize(q, s, o, fmt=fmt, group=grp, global_scale=gs)
            torch.cuda.synchronize()
            ce_24l_d2h = measure_stream(eng_24d.freeze().run, stream, reps=reps)[1]
            ce_24l_h2d = measure_stream(eng_24h.freeze().run, stream, reps=reps)[1]
        except Exception as e:
            ce_24l_d2h = f"ERR: {type(e).__name__}"

        rows.append({
            "format": fmt,
            "group": grp,
            "compressed_mb": comp_mb,
            "compile_1t_d2h_us": comp_1t_d2h,
            "compile_1t_quant_only_us": comp_1t_quant_only,
            "plain_1t_d2h_us": plain_d2h,
            "compile_1t_d2h_suspect_cpu_work": comp_suspect,
            "compile_error": comp_err,
            "ce_1t_d2h_us": ce_1t_d2h,
            "ce_1t_d2h_strat": strat_1t_d2h,
            "ce_24l_d2h_us": ce_24l_d2h,
            "compile_1t_h2d_us": comp_1t_h2d,
            "ce_1t_h2d_us": ce_1t_h2d,
            "ce_1t_h2d_strat": strat_1t_h2d,
            "ce_24l_h2d_us": ce_24l_h2d,
        })
        del hq_24l, hs_24l, x_1t, hq_1t, hs_1t, out_1t
        cleanup_cuda()

    # Also test Paged GPU -> Scattered Host Pool INT4 offload (if supported, e.g. v0.4.2+ / v0.5+)
    paged_offload = None
    try:
        kv_p = [torch.randn((24, S, 2 * h, D), dtype=torch.bfloat16, device=DEV) for h in heads]
        hq_p = [torch.zeros((32, S, 2 * h, 64), dtype=torch.uint8).pin_memory() for h in heads]
        hs_p = [torch.zeros((32, S, 2 * h, 1), dtype=torch.float32).pin_memory() for h in heads]
        eng_p = ce.Engine()
        def p_step(kvs, hqs, hss, gp, hs):
            with eng_p.scope():
                for x, q, s in zip(kvs, hqs, hss):
                    eng_p.quantize(x, q, s, fmt="int4", group=128, x_index=gp, index=hs)
        pl = eng_p.plan(p_step)
        gp0 = np.arange(8, dtype=np.int64)
        hs0 = np.array([5, 19, 2, 27, 11, 30, 7, 14], dtype=np.int64)
        for i in range(60):
            gp = np.random.default_rng(100 + i).permutation(24)[:8]
            hs = np.random.default_rng(200 + i).permutation(32)[:8]
            with torch.cuda.stream(stream):
                pl(kv_p, hq_p, hs_p, gp, hs)
        torch.cuda.synchronize()
        paged_offload = {
            "int4_24l_paged_offload_us": measure_stream(lambda: pl(kv_p, hq_p, hs_p, gp0, hs0), stream, reps=reps)[1],
            "strategy": eng_p.stats()["last"]["decisions"][0]["strategy"],
        }
        del kv_p, hq_p, hs_p
        cleanup_cuda()
    except Exception as e:
        paged_offload = {"status": "unsupported", "reason": str(e).splitlines()[0][:120]}

    del gpu_24l, out_24l
    cleanup_cuda()
    return {"raw_bf16_mb": raw_mb, "formats": rows, "paged_gpu_to_host_pool": paged_offload}


# ---------------------------------------------------------------------------
# Suite 10: 2D Box Block-FP8, Temp-Buffer Composition, & @torch.compile Hybrid Graph
# ---------------------------------------------------------------------------

def suite_box_and_compose(
    stream: torch.cuda.Stream,
    reps: int,
    with_compile: bool,
    caps: Dict[str, Any],
    **_,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    cq = get_quant_ref()

    # 10A: 2D (128, 128) Block FP8 Weight Quantization (16 experts x 2048 x 2048 = 134.22 MB)
    if caps.get("has_quant_block") and cq is not None:
        w_gpu = torch.randn((16, 2048, 2048), dtype=torch.bfloat16, device=DEV)
        w_pin = w_gpu.cpu().pin_memory()
        q_gpu = torch.empty((16, 2048, 2048), dtype=torch.float8_e4m3fn, device=DEV)
        s_gpu = torch.empty((16, 16, 16), dtype=torch.float32, device=DEV)
        w_out = torch.empty_like(w_gpu)

        eager_q = measure_stream(
            lambda: cq.native_quantize(w_gpu, q_gpu, s_gpu, None, "fp8_e4m3", 128, 1e-12, 1.0, None, None, (128, 128)),
            stream,
            reps=reps,
        )[1]
        eng_q, eng_dq, eng_hq = ce.Engine(), ce.Engine(), ce.Engine()
        with torch.cuda.stream(stream):
            for _ in range(60):
                eng_q.quantize(w_gpu, q_gpu, s_gpu, fmt="fp8_e4m3", block=(128, 128))
                eng_dq.dequantize(q_gpu, s_gpu, w_out, fmt="fp8_e4m3", block=(128, 128))
                eng_hq.quantize(w_pin, q_gpu, s_gpu, fmt="fp8_e4m3", block=(128, 128))
        torch.cuda.synchronize()
        out["box_128x128_fp8"] = {
            "eager_quant_us": eager_q,
            "ce_quant_us": measure_stream(eng_q.freeze().run, stream, reps=reps)[1],
            "ce_dequant_us": measure_stream(eng_dq.freeze().run, stream, reps=reps)[1],
            "ce_pinned_h2d_quant_us": measure_stream(eng_hq.freeze().run, stream, reps=reps)[1],
            "ce_pinned_h2d_strat": eng_hq.stats()["last"]["decisions"][0]["strategy"],
        }
        del w_gpu, w_pin, q_gpu, s_gpu, w_out
        cleanup_cuda()
    else:
        out["box_128x128_fp8"] = {"status": "skipped", "reason": "block=(bh, bw) unavailable"}

    # 10B: StepPlan Temporary-Buffer Composition (`src -> tmp -> dst`)
    if hasattr(ce, "plan"):
        L = 24
        srcs = [torch.randn((32, 4, 512, 128), dtype=torch.bfloat16, device=DEV) for _ in range(L)]
        dsts = [torch.zeros((32, 4, 512, 128), dtype=torch.bfloat16, device=DEV) for _ in range(L)]

        def step_tmp(ss, ds):
            for s, d in zip(ss, ds):
                tmp = torch.empty((4, 4, 512, 128), dtype=torch.bfloat16, device=DEV)
                tmp.copy_(s[2:6])
                d[10:14].copy_(tmp)

        eager_tmp_us = measure_stream(lambda: step_tmp(srcs, dsts), stream, reps=reps)[1]
        eng_t = ce.Engine()
        pl_t = eng_t.plan(step_tmp)
        with torch.cuda.stream(stream):
            for _ in range(120):
                pl_t(srcs, dsts)
        torch.cuda.synchronize()
        tr = next(iter(pl_t._layouts.values()))["traces"]
        replayable = bool(tr and tr[0].plain is None)
        ce_tmp_us = measure_stream(lambda: pl_t(srcs, dsts), stream, reps=reps)[1]
        out["temp_composition_24l"] = {
            "eager_with_temp_us": eager_tmp_us,
            "ce_step_plan_us": ce_tmp_us,
            "composed_replayable": replayable,
        }
        del srcs, dsts
        cleanup_cuda()

    # 10C: `@torch.compile` Hybrid Chunk-Prefill Step (`RMSNorm + Q/K/V + RoPE + KV scatter + FP8 quant + 24L state`)
    if with_compile and dynamo is not None and caps.get("has_compiled_stats") and cc() < (8, 9):
        out["compiled_hybrid_step"] = {"status": "skipped", "reason": "the step quantizes to fp8: Triton fp8 needs sm89+"}
    elif with_compile and dynamo is not None and caps.get("has_compiled_stats") and bf16_ok():
      try:
        reset_dynamo()
        T, Dm, Hq, Hkv, D, L = 1024, 2048, 16, 2, 128, 24
        B_pages, S_page = 128, 64
        h = torch.randn((T, Dm), dtype=torch.bfloat16, device=DEV)
        rms_w = torch.ones((Dm,), dtype=torch.bfloat16, device=DEV)
        wq = torch.randn((Dm, Hq * D), dtype=torch.bfloat16, device=DEV) * 0.02
        wk = torch.randn((Dm, Hkv * D), dtype=torch.bfloat16, device=DEV) * 0.02
        wv = torch.randn((Dm, Hkv * D), dtype=torch.bfloat16, device=DEV) * 0.02
        cos = torch.randn((T, 1, D // 2), dtype=torch.bfloat16, device=DEV)
        sin = torch.randn((T, 1, D // 2), dtype=torch.bfloat16, device=DEV)
        vc = torch.zeros((B_pages, Hkv, S_page, D), dtype=torch.bfloat16, device=DEV)
        kq = torch.zeros((B_pages, S_page, Hkv, D), dtype=torch.float8_e4m3fn, device=DEV)
        ks = torch.zeros((B_pages, S_page, Hkv, 1), dtype=torch.float32, device=DEV)
        st_s = [torch.randn((32, 256), dtype=torch.bfloat16, device=DEV) for _ in range(L)]
        st_d = [torch.empty_like(s) for s in st_s]
        pids = torch.randperm(B_pages, device=DEV)[:16]

        def rope(x, c, s):
            x1, x2 = x[..., : D // 2], x[..., D // 2 :]
            return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

        def step_torch(h_, w_r, w_q, w_k, w_v, c_, s_, vc_, kq_, ks_, ix_, sd_, ss_):
            n = h_ * torch.rsqrt(h_.float().pow(2).mean(-1, keepdim=True) + 1e-6).bfloat16() * w_r
            q = rope((n @ w_q).view(T, Hq, D), c_, s_)
            k = rope((n @ w_k).view(T, Hkv, D), c_, s_)
            v = (n @ w_v).view(T, Hkv, D)
            vc_.index_copy_(0, ix_, v.reshape(16, S_page, Hkv, D).permute(0, 2, 1, 3))
            cq.native_quantize(k.view(16, S_page, Hkv, D), kq_, ks_, None, "fp8_e4m3", D, 1e-12, 1.0, None, ix_)
            for dt, st in zip(sd_, ss_):
                dt.copy_(st)
            return q

        def step_hybrid(h_, w_r, w_q, w_k, w_v, c_, s_, vc_, kq_, ks_, ix_, sd_, ss_):
            n = h_ * torch.rsqrt(h_.float().pow(2).mean(-1, keepdim=True) + 1e-6).bfloat16() * w_r
            q = rope((n @ w_q).view(T, Hq, D), c_, s_)
            k = rope((n @ w_k).view(T, Hkv, D), c_, s_)
            v = (n @ w_v).view(T, Hkv, D)
            v_lz = ce.lazy(v).reshape(16, S_page, Hkv, D).permute(0, 2, 1, 3)
            ce.copy_(ce.lazy(vc_).index_select(0, ix_), v_lz)
            ce.quantize(k.view(16, S_page, Hkv, D), kq_, ks_, fmt="fp8_e4m3", group=D, index=ix_)
            ce.copy_many(sd_, ss_)
            return q

        exp = dynamo.explain(step_hybrid)(h, rms_w, wq, wk, wv, cos, sin, vc, kq, ks, pids, st_d, st_s)
        dynamo.reset()
        c_pure = torch.compile(step_torch)
        with torch.cuda.stream(stream):
            for _ in range(5):
                c_pure(h, rms_w, wq, wk, wv, cos, sin, vc, kq, ks, pids, st_d, st_s)
        torch.cuda.synchronize()

        dynamo.reset()
        c_hyb = torch.compile(step_hybrid)
        with torch.cuda.stream(stream):
            for _ in range(45):
                c_hyb(h, rms_w, wq, wk, wv, cos, sin, vc, kq, ks, pids, st_d, st_s)
        torch.cuda.synchronize()

        out["compiled_hybrid_step"] = {
            "dynamo_graph_count": exp.graph_count,
            "eager_us": measure_stream(lambda: step_torch(h, rms_w, wq, wk, wv, cos, sin, vc, kq, ks, pids, st_d, st_s), stream, reps=reps)[1],
            "torch_compile_alone_us": measure_stream(lambda: c_pure(h, rms_w, wq, wk, wv, cos, sin, vc, kq, ks, pids, st_d, st_s), stream, reps=reps)[1],
            "torch_compile_plus_ce_us": measure_stream(lambda: c_hyb(h, rms_w, wq, wk, wv, cos, sin, vc, kq, ks, pids, st_d, st_s), stream, reps=reps)[1],
            "compiled_baselines": ce.compiled_stats().get("baselines", []),
        }
        del h, rms_w, wq, wk, wv, cos, sin, vc, kq, ks, st_s, st_d, pids
        cleanup_cuda()
      except Exception as e:
        out["compiled_hybrid_step"] = {"status": "error", "reason": short_err(e)}
        cleanup_cuda()
    else:
        out["compiled_hybrid_step"] = {"status": "skipped", "reason": "compiled_stats or dynamo unavailable"}

    return out


# ---------------------------------------------------------------------------
# Suite 11: Quantization Formats (every format, quantize & dequantize, bits vs the recipe)
# ---------------------------------------------------------------------------

_Q_DTYPE = {"fp8_e4m3": "float8_e4m3fn", "mxfp8_e4m3": "float8_e4m3fn", "fp8_e5m2": "float8_e5m2",
            "mxfp8_e5m2": "float8_e5m2", "int8": "int8"}
_S_DTYPE = {"mxfp8_e4m3": "float8_e8m0fnu", "mxfp8_e5m2": "float8_e8m0fnu", "nvfp4": "float8_e4m3fn"}
_QUANT_FORMATS = [  # (format, group, block)
    ("fp8_e4m3", 128, None), ("int8", 128, None), ("int4", 128, None), ("uint4", 128, None),
    ("mxfp8_e4m3", 32, None), ("nvfp4", 16, None), ("fp8_e4m3", None, (128, 128)), ("int8", None, (128, 128)),
]


def _compiled_or_none(fn, *args):
    """fn compiled by Inductor, or None (e.g. FP8 where Triton has no FP8 for the GPU)."""
    import logging
    for name in ("torch._inductor", "torch._dynamo"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    try:
        c = torch.compile(fn)
        c(*args)
        torch.cuda.synchronize()
        return c
    except Exception:
        return None


def suite_quant_formats(stream: torch.cuda.Stream, reps: int, quick: bool, with_compile: bool, **_) -> Dict[str, Any]:
    q_ref = get_quant_ref()
    if q_ref is None or not hasattr(ce, "quantize"):
        return {"status": "skipped", "reason": "no quantization in this copyengine"}
    M, K = (2048, 4096) if quick else (8192, 4096)
    x = (torch.randn(M, K, device=DEV) * 4).to(torch.bfloat16 if bf16_ok() else torch.float32)
    out: List[Dict[str, Any]] = []
    for fmt, group, block in _QUANT_FORMATS:
        if fmt not in q_ref.FMT or any(not has_dtype(d) for d in (_Q_DTYPE.get(fmt), _S_DTYPE.get(fmt)) if d):
            out.append({"format": fmt, "status": "skipped", "reason": "format or dtype unavailable"})
            continue
        packed = fmt in q_ref.PACKED
        q = torch.zeros(M, K // 2 if packed else K, dtype=getattr(torch, _Q_DTYPE.get(fmt, "uint8")), device=DEV)
        if block:
            bh, g = block
            scale = torch.zeros(M // bh, K // g, device=DEV)
        else:
            bh, g = 1, q_ref.UNIT.get(fmt, group)
            scale = torch.zeros(M, K // g, dtype=getattr(torch, _S_DTYPE.get(fmt, "float32")), device=DEV)
        zero = torch.zeros(M, K // g, dtype=torch.uint8, device=DEV) if fmt == "uint4" else None
        kw = {"fmt": fmt, "block": block} if block else {"fmt": fmt, "group": g}
        blk = (bh, g) if block else None
        row: Dict[str, Any] = {"format": fmt, "unit": f"block {block[0]}x{block[1]}" if block else f"group {g}",
                               "mb": round((x.nbytes + q.nbytes + scale.nbytes) / 2**20, 1)}
        try:
            e = ce.Engine()
            back = torch.empty_like(x)

            def ce_q():
                e.quantize(x, q, scale, zero, **kw)

            def ce_d():
                e.dequantize(q, scale, back, zero, **kw)

            def rq(x, q, scale, zero):
                q_ref.native_quantize(x, q, scale, zero, fmt, g, q_ref.EPS, 1.0, None, None, blk)

            def rd(q, scale, zero, back):
                q_ref.native_dequantize(q, scale, zero, back, fmt, g, 1.0, None, None, blk)
            for _ in range(40):  # copyengine explores and locks
                ce_q()
                ce_d()
            torch.cuda.synchronize()
            want_q, want_s = q.clone(), scale.clone()
            rq(x, want_q, want_s, None if zero is None else zero.clone())
            ce_q()
            torch.cuda.synchronize()
            row["bits_match_recipe"] = bool(torch.equal(q.view(torch.uint8), want_q.view(torch.uint8))
                                            and torch.equal(scale.view(torch.uint8), want_s.view(torch.uint8)))
            row["ce_quant_us"] = measure_stream(ce_q, stream, reps=reps)
            row["ce_dequant_us"] = measure_stream(ce_d, stream, reps=reps)
            row["torch_quant_us"] = measure_stream(lambda: rq(x, q, scale, zero), stream, reps=max(5, reps // 3))
            row["torch_dequant_us"] = measure_stream(lambda: rd(q, scale, zero, back), stream, reps=max(5, reps // 3))
            if with_compile:
                cq = _compiled_or_none(rq, x, q, scale, zero)
                cd = _compiled_or_none(rd, q, scale, zero, back)
                row["compiled_quant_us"] = measure_stream(lambda: cq(x, q, scale, zero), stream, reps=reps) if cq else "no compile"
                row["compiled_dequant_us"] = measure_stream(lambda: cd(q, scale, zero, back), stream, reps=reps) if cd else "no compile"
            row["strategies"] = sorted(set(e.stats()["last"]["strategies"]))
        except Exception as ex:
            row["error"] = short_err(ex)
        out.append(row)
        reset_dynamo()
    del x
    cleanup_cuda()
    return {"shape": [M, K], "formats": out}


# ---------------------------------------------------------------------------
# Suite 12: Small-Copy Call Overhead (gathers of 1-64 blocks; one copy call, split)
# ---------------------------------------------------------------------------

def _per_call(fn, calls: int) -> float:
    """Microseconds per call over back-to-back calls (the host's work per call included)."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(calls):
        fn()
    torch.cuda.synchronize()
    return round((time.perf_counter() - t0) / calls * 1e6, 2)


def suite_calls(stream: torch.cuda.Stream, reps: int, quick: bool, **_) -> Dict[str, Any]:
    calls = 200 if quick else 1000
    pool = torch.randn(4096, 1024, device=DEV)  # 4 KB blocks
    gathers = []
    for nb in (1, 8, 64):
        out = torch.empty(nb, 1024, device=DEV)
        idx = torch.randperm(4096, device=DEV)[:nb]
        il = idx.tolist()
        row: Dict[str, Any] = {"blocks": nb}

        def loop():
            for i, b in enumerate(il):
                out[i].copy_(pool[b])
        row["torch_loop_us"] = _per_call(loop, calls)
        row["torch_index_select_us"] = _per_call(lambda: torch.index_select(pool, 0, idx, out=out), calls)
        try:
            e = ce.Engine()
            for _ in range(60):
                e.gather(pool, idx, 0, out=out)
            row["ce_gather_us"] = _per_call(lambda: e.gather(pool, idx, 0, out=out), calls)
            if hasattr(e, "plan"):
                step = e.plan(lambda pool, idx, out: e.gather(pool, idx, 0, out=out))
                for _ in range(60):
                    step(pool, idx, out)
                row["ce_step_plan_us"] = _per_call(lambda: step(pool, idx, out), calls)
        except Exception as ex:
            row["error"] = short_err(ex)
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                torch.index_select(pool, 0, idx, out=out)
            row["torch_graph_us"] = _per_call(g.replay, calls)
        except Exception as ex:
            row["torch_graph_us"] = short_err(ex)
        row["correct"] = bool(torch.equal(out, pool[idx]))
        gathers.append(row)

    # One copy of a 32 KB block per call, 2048 calls: where the time goes.
    n = 2048
    cache = torch.zeros(n, 16, 8, 128, dtype=torch.bfloat16, device=DEV)
    slot = torch.randn(n, 16, 8, 128).to(torch.bfloat16).pin_memory()
    dsts, srcs = [cache[i] for i in range(n)], [slot[i] for i in range(n)]
    split: Dict[str, Any] = {}

    def batch_us(batch):
        return round(_per_call(batch, 5) / n, 2)

    def views():
        for i in range(n):
            cache[i], slot[i]
    split["view_us"] = batch_us(views)

    def torch_copy():
        for d, s in zip(dsts, srcs):
            d.copy_(s, non_blocking=True)
    split["torch_copy_us"] = batch_us(torch_copy)
    alias = torch.ops.aten.alias.default

    def dispatch():
        for d in dsts:
            alias(d)
    split["torch_op_dispatch_us"] = batch_us(dispatch)
    try:
        e = ce.Engine()

        def ce_scope():
            with e.scope():
                for d, s in zip(dsts, srcs):
                    e.copy_(d, s)
        for _ in range(20):
            ce_scope()
        split["ce_copy_in_scope_us"] = batch_us(ce_scope)

        def intercepted():
            with e.scope():
                for d, s in zip(dsts, srcs):
                    d.copy_(s, non_blocking=True)
        split["intercepted_copy_in_scope_us"] = batch_us(intercepted)
        ops = getattr(getattr(ce, "_torch_ops", None), "copy_", None)
        if ops is not None:
            h = e._handle

            def raw():
                e._e.begin_scope()
                for d, s in zip(dsts, srcs):
                    ops(d, s, h)
                e._e.end_scope()
            split["ce_custom_op_queued_us"] = batch_us(raw)
    except Exception as ex:
        split["error"] = short_err(ex)
    del cache, slot, pool
    cleanup_cuda()
    return {"gathers": gathers, "one_copy_split": split}


# ---------------------------------------------------------------------------
# Suite 13: Peer-to-Peer (two GPUs: KV migration, TP resharding, converting migration)
# ---------------------------------------------------------------------------

def suite_p2p(stream: torch.cuda.Stream, reps: int, quick: bool, **_) -> Dict[str, Any]:
    if torch.cuda.device_count() < 2:
        return {"status": "skipped", "reason": "needs two GPUs"}

    def sync_all():
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    def timed(fn):
        for _ in range(3):
            fn()
        sync_all()
        ts = []
        for _ in range(reps):
            sync_all()
            t0 = time.perf_counter()
            fn()
            sync_all()
            ts.append(time.perf_counter() - t0)
        return round(statistics.median(ts) * 1e6, 1)

    def tuned(body):
        e = ce.Engine()
        for _ in range(60):
            body(e)
        sync_all()
        return e

    L = 8 if quick else 32
    rows = []
    for a, b in ((0, 1), (1, 0)):
        row: Dict[str, Any] = {"pair": f"{a}->{b}", "peer_access": torch.cuda.can_device_access_peer(a, b)}
        try:  # KV migration: blocks of a paged pool on GPU a, picked by index, into GPU b
            NB, B = 512, 64
            pool = torch.randn(L, NB, 16, 8, 128, dtype=torch.bfloat16, device=f"cuda:{a}")
            ids = torch.randperm(NB, device=f"cuda:{a}")[:B]
            out = torch.zeros(L, B, 16, 8, 128, dtype=torch.bfloat16, device=f"cuda:{b}")
            dsts = [out[l, i] for l in range(L) for i in range(B)]
            srcs = [pool[l, x] for l in range(L) for x in ids.tolist()]
            row["kv_torch_foreach_us"] = timed(lambda: torch._foreach_copy_(dsts, srcs, non_blocking=True))
            row["kv_torch_index_select_us"] = timed(lambda: out.copy_(pool.index_select(1, ids), non_blocking=True))
            e = tuned(lambda e: e.gather(pool, ids, 1, out=out))
            row["kv_ce_gather_us"] = timed(lambda: e.gather(pool, ids, 1, out=out))
            sync_all()
            row["kv_correct"] = bool(torch.equal(out.cpu(), pool[:, ids].cpu()))
            row["kv_strategies"] = sorted(set(e.stats()["last"]["strategies"]))
            del pool, out, dsts, srcs
        except Exception as ex:
            row["kv_error"] = short_err(ex)
        try:  # TP resharding: one column shard of every layer's weight to GPU b
            ws = [torch.randn(4096, 4096, dtype=torch.bfloat16, device=f"cuda:{a}") for _ in range(L)]
            shards = [w[:, 1024:2048] for w in ws]
            outs = [torch.empty(4096, 1024, dtype=torch.bfloat16, device=f"cuda:{b}") for _ in range(L)]
            row["reshard_torch_foreach_us"] = timed(lambda: torch._foreach_copy_(outs, shards, non_blocking=True))
            e = tuned(lambda e: e.copy_many(outs, shards))
            row["reshard_ce_copy_many_us"] = timed(lambda: e.copy_many(outs, shards))
            del ws, shards, outs
        except Exception as ex:
            row["reshard_error"] = short_err(ex)
        try:  # converting migration: f32 on GPU a to bf16 on GPU b
            src = torch.randn(1 << 24, device=f"cuda:{a}")
            dst = torch.empty(1 << 24, dtype=torch.bfloat16, device=f"cuda:{b}")
            row["convert_torch_us"] = timed(lambda: dst.copy_(src))
            e = tuned(lambda e: e.copy_(dst, src))
            row["convert_ce_us"] = timed(lambda: e.copy_(dst, src))
            del src, dst
        except Exception as ex:
            row["convert_error"] = short_err(ex)
        rows.append(row)
        cleanup_cuda()
    return {"pairs": rows}


# ---------------------------------------------------------------------------
# Multi-Version Comparison Printer & CLI Driver
# ---------------------------------------------------------------------------

SUITES: Dict[str, Callable[..., Dict[str, Any]]] = {
    "probe": suite_probe,
    "sweep": suite_sweep,
    "paged_kv": suite_paged_kv,
    "reblock": suite_reblock,
    "reshard": suite_reshard,
    "tma_sm_cap": suite_tma_sm_cap,
    "interference": suite_interference,
    "tile_quant": suite_tile_quant,
    "pcie_quant": suite_pcie_quant,
    "box_and_compose": suite_box_and_compose,
    "quant_formats": suite_quant_formats,
    "calls": suite_calls,
    "p2p": suite_p2p,
}


def print_compare(paths: List[str]) -> None:
    reports = [json.load(open(p)) for p in paths]
    vers = [r["capabilities"]["copyengine_version"] for r in reports]
    print(f"Comparing {len(reports)} reports: " + " vs ".join(vers))

    def fmt_pair(v):
        if isinstance(v, list) and len(v) == 2:
            return f"{v[1]:.1f} us"
        if isinstance(v, (int, float)):
            return f"{v:.1f} us"
        return str(v)

    def row(label: str, extractor: Callable[[dict], Any]):
        vals = [fmt_pair(extractor(r)) for r in reports]
        print(f"{label:44s} | " + " | ".join(f"{v:>14s}" for v in vals))

    hdr = f"{'Metric':44s} | " + " | ".join(f"{v:>14s}" for v in vers)
    print(hdr)
    print("-" * len(hdr))
    row("Paged 56L Pinned H2D 16B (StepPlan)", lambda r: next((x["ce_step_plan_dyn_us"] for x in r.get("suites", {}).get("paged_kv", {}).get("results", []) if x["host_kind"] == "pinned" and x["direction"] == "H2D" and x["blocks"] == 16), "—"))
    row("Paged 56L Pageable H2D 16B (StepPlan)", lambda r: next((x["ce_step_plan_dyn_us"] for x in r.get("suites", {}).get("paged_kv", {}).get("results", []) if x["host_kind"] == "pageable" and x["direction"] == "H2D" and x["blocks"] == 16), "—"))
    row("Paged 56L Pageable H2D 16B (torch foreach)", lambda r: next((x.get("torch_foreach_dyn_us", x.get("torch_foreach_us")) for x in r.get("suites", {}).get("paged_kv", {}).get("results", []) if x["host_kind"] == "pageable" and x["direction"] == "H2D" and x["blocks"] == 16), "—"))
    row("Reblock 56L D2D StepPlan (143 MB)", lambda r: r.get("suites", {}).get("reblock", {}).get("d2d_ce_step_plan_us", "—"))
    row("Reshard 24L TP2->TP4 D2D Frozen", lambda r: r.get("suites", {}).get("reshard", {}).get("d2d_ce_frozen_us", "—"))
    row("Narrow-Row 134MB Permute (SM=8)", lambda r: next((x.get("us", x.get("error")) for x in r.get("suites", {}).get("tma_sm_cap", {}).get("narrow_row_permute_134mb_by_sm", []) if x["max_sms"] == 8), "—"))
    row("24L Tile Quant fp8_e4m3 (134 MB)", lambda r: next((x["ce_quant_us"] for x in r.get("suites", {}).get("tile_quant", {}).get("formats", []) if x["format"] == "fp8_e4m3"), "—"))
    row("24L Tile Quant int4 (134 MB)", lambda r: next((x["ce_quant_us"] for x in r.get("suites", {}).get("tile_quant", {}).get("formats", []) if x["format"] == "int4"), "—"))
    row("24L Tile Quant nvfp4 (134 MB)", lambda r: next((x["ce_quant_us"] for x in r.get("suites", {}).get("tile_quant", {}).get("formats", []) if x["format"] == "nvfp4"), "—"))
    row("PCIe 1T Offload D2H int4 (134->36 MB)", lambda r: next((x["ce_1t_d2h_us"] for x in r.get("suites", {}).get("pcie_quant", {}).get("formats", []) if x["format"] == "int4"), "—"))
    row("Box (128,128) FP8 GPU Quant (134 MB)", lambda r: r.get("suites", {}).get("box_and_compose", {}).get("box_128x128_fp8", {}).get("ce_quant_us", "—"))
    row("@torch.compile + CE Hybrid Step", lambda r: r.get("suites", {}).get("box_and_compose", {}).get("compiled_hybrid_step", {}).get("torch_compile_plus_ce_us", "—"))

    def fmt_row(fmt, unit, key):
        return lambda r: next((x.get(key, "—") for x in r.get("suites", {}).get("quant_formats", {}).get("formats", [])
                               if x.get("format") == fmt and x.get("unit") == unit), "—")
    row("Quant fp8_e4m3 group 128 (quantize)", fmt_row("fp8_e4m3", "group 128", "ce_quant_us"))
    row("Quant fp8_e4m3 block 128x128 (quantize)", fmt_row("fp8_e4m3", "block 128x128", "ce_quant_us"))
    row("Quant uint4 group 128 (dequantize)", fmt_row("uint4", "group 128", "ce_dequant_us"))
    row("Gather 8 x 4 KB (step plan, per call)", lambda r: next((x.get("ce_step_plan_us", "—") for x in r.get("suites", {}).get("calls", {}).get("gathers", []) if x.get("blocks") == 8), "—"))
    row("One 32 KB copy in a scope (per call)", lambda r: r.get("suites", {}).get("calls", {}).get("one_copy_split", {}).get("ce_copy_in_scope_us", "—"))
    row("P2P KV migration 0->1 (ce gather)", lambda r: next((x.get("kv_ce_gather_us", "—") for x in r.get("suites", {}).get("p2p", {}).get("pairs", []) if x.get("pair") == "0->1"), "—"))


def main() -> None:
    ap = argparse.ArgumentParser(description="copyengine on its target workloads, one suite per kind")
    ap.add_argument("--suites", default="all", help="Comma-separated suite names or 'all'")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--max-sms", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--no-flush-l2", action="store_true", help="Do not flush L2 between repetitions")
    ap.add_argument("--json", default=None, help="Output JSON report path")
    ap.add_argument("--compare", nargs="+", default=None, help="Compare two or more saved JSON reports")
    args = ap.parse_args()

    if args.compare:
        print_compare(args.compare)
        return

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU is required")
    global FLUSH_L2
    FLUSH_L2 = not args.no_flush_l2

    caps = detect_capabilities()
    print(
        f"copy_bench: device={caps['device_name']} (sm_{caps['compute_capability']}) | "
        f"torch={caps['torch_version']} | copyengine={caps['copyengine_version']} | "
        f"bulk={caps['cuda_bulk']} | tma2d={caps['cuda_tma2d']}",
        flush=True,
    )

    chosen = list(SUITES.keys()) if args.suites == "all" else [s.strip() for s in args.suites.split(",") if s.strip()]
    for name in chosen:
        if name not in SUITES:
            sys.exit(f"Unknown suite {name!r}; available: {list(SUITES.keys())}")

    stream = torch.cuda.Stream()
    rooflines = suite_probe(stream, args.reps, args.quick) if "probe" in chosen or "sweep" in chosen else {}

    report: Dict[str, Any] = {
        "capabilities": caps,
        "suites": {},
    }

    for name in chosen:
        if name == "probe":
            report["suites"]["probe"] = rooflines
            print("[probe]", json.dumps(rooflines), flush=True)
            continue
        print(f"=== Running suite: {name} ===", flush=True)
        t0 = time.perf_counter()
        try:
            res = SUITES[name](
                stream=stream,
                reps=args.reps,
                quick=args.quick,
                with_compile=not args.skip_compile,
                max_sms=args.max_sms,
                rooflines=rooflines,
                caps=caps,
            )
            status = "done"
        except torch.cuda.OutOfMemoryError as e:
            res = {"status": "skipped", "reason": "out of GPU memory: " + short_err(e)}
            status = "skipped (OOM)"
        except Exception as e:
            import traceback
            traceback.print_exc()
            res = {"status": "error", "reason": short_err(e)}
            status = "error: " + short_err(e)
        reset_dynamo()
        try:
            cleanup_cuda()
        except Exception as e:  # a sticky CUDA error: later suites cannot run in this process
            report["suites"][name] = res
            print(f"[{name}] {status}; CUDA context unusable ({short_err(e)}), stopping", flush=True)
            break
        report["suites"][name] = res
        print(f"[{name}] {status} in {time.perf_counter() - t0:.0f} s", flush=True)
        if args.json:  # partial results survive a crash or a Colab disconnect
            with open(args.json, "w") as f:
                json.dump(report, f, indent=2, default=str)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Saved JSON report to {args.json}", flush=True)


if __name__ == "__main__":
    main()
