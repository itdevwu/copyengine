"""copyengine vs torch on the copy patterns copyengine targets.

    python bench/bench_torch.py [--json out.json] [--quick]

Prints median microseconds per iteration. Results are machine specific; keep
them out of the repository.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch

import copyengine as ce

DEV = "cuda"


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


def tuned_engine(body, rounds: int = 24) -> ce.Engine:
    """An engine whose buckets for `body` are explored and locked."""
    e = ce.Engine(tune=True, tune_trials=3, tune_topk=4)
    for _ in range(rounds):
        body(e)
    torch.cuda.synchronize()
    return e


def run_case(name, check, torch_variants: dict, ce_body, reps: int, ce_api=None) -> dict:
    row = {"case": name}
    for k, fn in torch_variants.items():
        row[k] = timeit(fn, reps)
    if ce_api is not None:   # explicit vectorized API: one call with offset arrays
        ea = tuned_engine(ce_api)
        row["ce_api"] = timeit(lambda: ce_api(ea), reps)
        check()
    e = tuned_engine(ce_body)
    row["ce"] = timeit(lambda: ce_body(e), reps)
    frozen = e.freeze()
    row["ce_frozen"] = timeit(frozen.run, reps)
    row["strategies"] = sorted(set(e.stats()["last"]["strategies"]))
    row["calls"] = e.stats()["last"]["calls"]
    check()
    return row


# ---- cases ---------------------------------------------------------------------------------


def case_layer_block_h2d(reps, L=32, B=64, T=16, H=8, D=128):
    """PD / offload load: per (layer, block) H2D copies from a pinned slot into a paged KV cache."""
    slot = torch.randn(L, B, T, H, D, dtype=torch.bfloat16).pin_memory()
    cache = torch.zeros(L, 4 * B, T, H, D, dtype=torch.bfloat16, device=DEV)
    ids = torch.randperm(4 * B)[:B]
    idl = ids.tolist()
    ids_dev = ids.to(DEV)

    def loop():
        for l in range(L):
            for i, b in enumerate(idl):
                cache[l, b].copy_(slot[l, i], non_blocking=True)

    def vectorized():
        for l in range(L):
            cache[l].index_copy_(0, ids_dev, slot[l].to(DEV, non_blocking=True))

    def body(e):
        with e.scope():
            for l in range(L):
                for i, b in enumerate(idl):
                    e.copy_(cache[l, b], slot[l, i])

    def check():
        torch.testing.assert_close(cache[:, ids].cpu(), slot, rtol=0, atol=0)

    blk = T * H * D
    l_idx = torch.arange(L).repeat_interleave(B)
    i_idx = torch.arange(B).repeat(L)
    src_off = ((l_idx * B + i_idx) * blk).numpy()
    dst_off = ((l_idx * 4 * B + ids.repeat(L)) * blk).numpy()
    length = [blk] * (L * B)

    def api(e):
        e.segments(slot, cache, src_off, dst_off, length)

    return run_case("layer_block_h2d", check, {"torch_loop": loop, "torch_vec": vectorized}, body, reps, api)


def case_qscale_expand(reps, L=32, B=64, N=16, S=8):
    """Tiny broadcast copies: scale[layer][blk] (N, S) <- k_qs[blk] (N,) expanded (design case 9)."""
    k_qs = torch.rand(B, N, 1).pin_memory()
    scale = torch.zeros(L, B, N, S, device=DEV)

    def loop():
        for l in range(L):
            for b in range(B):
                scale[l, b].copy_(k_qs[b].expand(N, S), non_blocking=True)

    def vectorized():
        scale.copy_(k_qs.to(DEV, non_blocking=True).expand(B, N, S).unsqueeze(0).expand(L, B, N, S))

    def body(e):
        with e.scope():
            for l in range(L):
                for b in range(B):
                    e.copy_(scale[l, b], k_qs[b].expand(N, S))

    def check():
        torch.testing.assert_close(scale.cpu(), k_qs.expand(B, N, S).unsqueeze(0).expand(L, B, N, S), rtol=0, atol=0)

    return run_case("qscale_expand", check, {"torch_loop": loop, "torch_vec": vectorized}, body, reps)


def case_gather_permute_cast(reps, NB=2048, T=16, H=8, D=128, n=512):
    """D2D: gather blocks, permute [blk, tok, head, dim] -> [blk, head, tok, dim], cast f32 -> f16."""
    cache = torch.randn(NB, T, H, D, device=DEV)
    ids = torch.randint(0, NB, (n,))
    ids_dev = ids.to(DEV)
    out = torch.empty(n, H, T, D, dtype=torch.float16, device=DEV)
    src = ce.lazy(cache).index_select(0, ids).permute(0, 2, 1, 3)

    def torch_vec():
        out.copy_(cache.index_select(0, ids_dev).permute(0, 2, 1, 3))

    def body(e):
        e.copy_(out, src)

    def check():
        torch.testing.assert_close(out, cache[ids_dev].permute(0, 2, 1, 3).half(), rtol=0, atol=0)

    return run_case("gather_permute_cast_d2d", check, {"torch_vec": torch_vec}, body, reps)


def case_dequant(reps, M=4096, K=4096):
    """int8 -> f16 with a scale: (q.float() * s).half()."""
    q = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=DEV)
    out = torch.empty(M, K, dtype=torch.float16, device=DEV)
    s = 0.0125
    fn = ce.Elementwise(scale=s)

    def torch_vec():
        out.copy_(q.float() * s)

    def body(e):
        e.copy_(out, q, fn=fn)

    def check():
        torch.testing.assert_close(out, (q.float() * s).half(), rtol=0, atol=0)

    return run_case("dequant_int8_f16", check, {"torch_vec": torch_vec}, body, reps)


def case_permute_h2d(reps, A=4096, B=8, D=128, pinned=True):
    """Dense host permute then H2D: torch reorders on the CPU; ce moves once and permutes on the GPU."""
    x = torch.randn(A, B, D, dtype=torch.float16)
    x = x.pin_memory() if pinned else x
    src = x.permute(1, 0, 2)
    dst = torch.empty(B, A, D, dtype=torch.float16, device=DEV)

    def torch_vec():
        dst.copy_(src, non_blocking=True)

    def body(e):
        e.copy_(dst, src)

    def check():
        torch.testing.assert_close(dst.cpu(), src.contiguous(), rtol=0, atol=0)

    tag = "pinned" if pinned else "pageable"
    return run_case(f"permute_h2d_{tag}", check, {"torch_vec": torch_vec}, body, reps)


def case_offload_d2h(reps, NB=4096, T=16, H=8, D=128, n=1024, pinned=True):
    """Offload: scattered device blocks -> contiguous host buffer."""
    cache = torch.randn(NB, T, H, D, dtype=torch.bfloat16, device=DEV)
    ids = torch.randperm(NB)[:n]
    idl = ids.tolist()
    ids_dev = ids.to(DEV)
    host = torch.empty(n, T, H, D, dtype=torch.bfloat16)
    host = host.pin_memory() if pinned else host

    def loop():
        for i, b in enumerate(idl):
            host[i].copy_(cache[b], non_blocking=True)

    def torch_vec():
        host.copy_(cache.index_select(0, ids_dev), non_blocking=True)

    def body(e):
        with e.scope():
            for i, b in enumerate(idl):
                e.copy_(host[i], cache[b])

    def check():
        torch.testing.assert_close(host, cache[ids_dev].cpu(), rtol=0, atol=0)

    blk = T * H * D

    def api(e):
        e.segments(cache, host, ids.numpy() * blk, torch.arange(n).numpy() * blk, [blk] * n)

    tag = "pinned" if pinned else "pageable"
    return run_case(f"offload_d2h_{tag}", check, {"torch_loop": loop, "torch_vec": torch_vec}, body, reps, api)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write results here")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs a CUDA GPU", file=sys.stderr)
        return 1
    rows = [
        case_layer_block_h2d(args.reps),
        case_qscale_expand(args.reps),
        case_gather_permute_cast(args.reps),
        case_dequant(args.reps),
        case_permute_h2d(args.reps, pinned=True),
        case_permute_h2d(args.reps, pinned=False),
        case_offload_d2h(args.reps, pinned=True),
        case_offload_d2h(args.reps, pinned=False),
    ]
    cols = ["torch_loop", "torch_vec", "ce", "ce_api", "ce_frozen"]
    print(f"{'case':26s}" + "".join(f"{c:>13s}" for c in cols) + "   best-torch/ce  strategies (calls)")
    for r in rows:
        best = min(r[c] for c in ("torch_loop", "torch_vec") if c in r)
        cells = "".join(f"{r[c]:13.1f}" if c in r else f"{'-':>13s}" for c in cols)
        ce_best = min(r[c] for c in ("ce", "ce_api") if c in r)
        print(f"{r['case']:26s}{cells}   {best / ce_best:12.2f}x  {','.join(r['strategies'])} ({r['calls']})")
    out = {"hardware": ce.hardware_fingerprint(), "torch": torch.__version__, "copyengine": ce.__version__,
           "unit": "us", "rows": rows}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
