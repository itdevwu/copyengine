"""copyengine vs torch on copies between two GPUs (needs two GPUs).

    python bench/bench_p2p.py [--json out.json] [--quick] [--pair 0,1]

Cases: KV-cache migration (blocks gathered from one GPU's paged pool into
another GPU), tensor-parallel resharding (column shards of every layer's weight
to another GPU), and a converting migration (f32 to bf16 on the other GPU).
Prints median microseconds per iteration. Results are machine specific; keep
them out of the repository.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

import copyengine as ce


def sync_all():
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def timeit(fn, reps: int) -> float:
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
    return statistics.median(ts) * 1e6


def tuned(body, rounds=80):
    e = ce.Engine(tune_trials=3, tune_topk=4)
    for _ in range(rounds):
        body(e)
    sync_all()
    return e


def graphed(fn, device):
    """fn captured once in a CUDA graph on `device`, replayed; None where capture fails."""
    try:
        with torch.cuda.device(device):
            fn()
            sync_all()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
        return g.replay
    except Exception:  # cross-device copies may not be capturable
        return None


def run_case(name, check, torch_variants, ce_body, reps, device):
    row = {"case": name}
    for k, fn in torch_variants.items():
        if fn is not None:
            row[k] = timeit(fn, reps)
    e = tuned(ce_body)
    row["ce"] = timeit(lambda: ce_body(e), reps)
    check()
    try:
        fr = e.freeze()
        row["ce_frozen"] = timeit(fr.run, reps)
        check()
    except RuntimeError:
        pass
    last = e.stats()["last"]
    row["strategies"] = sorted(set(last["strategies"]))
    row["calls"] = last["calls"]
    return row


def case_kv_migrate(a, b, reps, L=32, NB=512, B=64, T=16, H=8, D=128):
    """Blocks of a paged KV cache on GPU a, picked by index, into a contiguous buffer on GPU b."""
    pool = torch.randn(L, NB, T, H, D, dtype=torch.bfloat16, device=f"cuda:{a}")
    ids = torch.randperm(NB, device=f"cuda:{a}")[:B]
    idl = ids.tolist()
    out = torch.zeros(L, B, T, H, D, dtype=torch.bfloat16, device=f"cuda:{b}")
    dsts = [out[l, i] for l in range(L) for i in range(B)]
    srcs = [pool[l, x] for l in range(L) for x in idl]

    def loop():
        for d, s in zip(dsts, srcs):
            d.copy_(s, non_blocking=True)

    def gather_then_copy():
        out.copy_(pool.index_select(1, ids), non_blocking=True)

    def check():
        sync_all()
        assert torch.equal(out.cpu(), pool[:, ids].cpu())

    variants = {"torch_loop": loop, "torch_foreach": lambda: torch._foreach_copy_(dsts, srcs, non_blocking=True),
                "torch_index_select": gather_then_copy, "torch_loop_graph": graphed(loop, b)}
    return run_case(f"kv_migrate {a}->{b} L={L} B={B}", check, variants,
                    lambda e: e.gather(pool, ids, 1, out=out), reps, b)


def case_tp_reshard(a, b, reps, L=24, K=4096, N=4096, parts=4):
    """Every layer's weight [K, N] on GPU a: one column shard each to GPU b."""
    ws = [torch.randn(K, N, dtype=torch.bfloat16, device=f"cuda:{a}") for _ in range(L)]
    w = N // parts
    shards = [x[:, w:2 * w] for x in ws]
    outs = [torch.empty(K, w, dtype=torch.bfloat16, device=f"cuda:{b}") for _ in range(L)]

    def loop():
        for d, s in zip(outs, shards):
            d.copy_(s, non_blocking=True)

    def check():
        sync_all()
        for d, s in zip(outs, shards):
            assert torch.equal(d.cpu(), s.cpu())

    variants = {"torch_loop": loop, "torch_foreach": lambda: torch._foreach_copy_(outs, shards, non_blocking=True)}
    return run_case(f"tp_reshard {a}->{b} L={L} {K}x{N}/{parts}", check, variants,
                    lambda e: e.copy_many(outs, shards), reps, b)


def case_convert_migrate(a, b, reps, n=1 << 24):
    """f32 on GPU a to bf16 on GPU b."""
    src = torch.randn(n, device=f"cuda:{a}")
    dst = torch.empty(n, dtype=torch.bfloat16, device=f"cuda:{b}")

    def check():
        sync_all()
        want = torch.empty_like(dst)
        want.copy_(src)
        assert torch.equal(dst.view(torch.int16).cpu(), want.view(torch.int16).cpu())

    return run_case(f"convert_migrate {a}->{b} f32->bf16 n={n}", check, {"torch": lambda: dst.copy_(src)},
                    lambda e: e.copy_(dst, src), reps, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pair", default="0,1")
    args = ap.parse_args()
    if torch.cuda.device_count() < 2:
        print("needs two CUDA GPUs")
        return
    a, b = (int(x) for x in args.pair.split(","))
    reps = 10 if args.quick else 40
    print(f"GPU {a} -> GPU {b}, peer access: {torch.cuda.can_device_access_peer(a, b)}")
    rows = []
    for case in (case_kv_migrate, case_tp_reshard, case_convert_migrate):
        for x, y in ((a, b), (b, a)):
            try:
                row = case(x, y, reps)
            except Exception as ex:  # a case that fails is reported, the rest still run
                row = {"case": f"{case.__name__} {x}->{y}", "error": f"{type(ex).__name__}: {ex}"}
            rows.append(row)
            times = {k: v for k, v in row.items() if isinstance(v, float)}
            best_torch = min((v for k, v in times.items() if k.startswith("torch")), default=None)
            print(row["case"], " ".join(f"{k}={v:.0f}us" for k, v in times.items()),
                  f"speedup={best_torch / row['ce']:.2f}x" if best_torch and "ce" in row else row.get("error", ""),
                  row.get("strategies", ""), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
