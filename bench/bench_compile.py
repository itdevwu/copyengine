"""copyengine vs torch.compile on the copy patterns copyengine targets.

    python bench/bench_compile.py [--json out.json] [--quick] [--only NAME ...]

Each case runs three ways after warmup: torch eager, the same function under
torch.compile (default mode and, where the tensors allow CUDA graphs,
"reduce-overhead"), and copyengine (explored and locked first, as after a
model's warmup). Prints median microseconds per iteration (host time to
completion, synchronized). Results are machine specific; keep them out of
the repository.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch

import copyengine as ce
from copyengine import _quant

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


class _NoScope:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _scope(e):
    """A scope eagerly; inside a compiled region each call is its own graph node (scopes stay outside)."""
    return _NoScope() if torch.compiler.is_compiling() else e.scope()


def compiled(fn, mode=None):
    torch._dynamo.reset()
    c = torch.compile(fn, mode=mode, fullgraph=False, dynamic=False)
    return c


def warm(fn, rounds: int):
    for _ in range(rounds):
        fn()
    torch.cuda.synchronize()


# ---- cases: (eager function, copyengine function given an engine, check) -----------------

def case_offload(q):
    """Paged KV offload: gather pages of every layer into pinned host buffers."""
    L, blocks, tok, hd, n = (8 if q else 24), 256, 64, 1024, 64
    kv = [torch.randn(blocks, tok, hd, device=DEV).bfloat16() for _ in range(L)]
    host = [torch.empty(n, tok, hd, dtype=torch.bfloat16).pin_memory() for _ in range(L)]
    idx = torch.randperm(blocks, device=DEV)[:n]  # block ids on the GPU, for both

    def eager():
        for l in range(L):
            host[l].copy_(kv[l].index_select(0, idx), non_blocking=True)

    def ce_fn(e):
        with _scope(e):
            for l in range(L):
                e.copy_(host[l], ce.lazy(kv[l]).index_select(0, idx))

    def check():
        return all(torch.equal(host[l], kv[l].index_select(0, idx).cpu()) for l in range(L))

    return eager, ce_fn, check, L * n * tok * hd * 2, False


def case_onload(q):
    """Paged KV load: pinned host pages into scattered GPU blocks."""
    L, blocks, tok, hd, n = (8 if q else 24), 256, 64, 1024, 64
    kv = [torch.zeros(blocks, tok, hd, device=DEV, dtype=torch.bfloat16) for _ in range(L)]
    host = [torch.randn(n, tok, hd).bfloat16().pin_memory() for _ in range(L)]
    idx = torch.randperm(blocks, device=DEV)[:n]

    def eager():
        for l in range(L):
            kv[l].index_copy_(0, idx, host[l].to(DEV, non_blocking=True))

    def ce_fn(e):
        with _scope(e):
            for l in range(L):
                e.copy_(ce.lazy(kv[l]).index_select(0, idx), host[l])

    def check():
        return all(torch.equal(kv[l].index_select(0, idx).cpu(), host[l]) for l in range(L))

    return eager, ce_fn, check, L * n * tok * hd * 2, False


def case_many_small(q):
    """Many small device tensors (one per layer and kind): one copy each."""
    n, numel = (96 if q else 576), 32 * 1024
    src = [torch.randn(numel, device=DEV) for _ in range(n)]
    dst = [torch.empty_like(s) for s in src]

    def eager():
        for d, s in zip(dst, src):
            d.copy_(s)

    def ce_fn(e):
        e.copy_many(dst, src)

    def check():
        return all(torch.equal(d, s) for d, s in zip(dst, src))

    return eager, ce_fn, check, n * numel * 4, True


def case_transpose_cast(q):
    """[B, S, H, D] -> [B, H, S, D] with a cast to fp16 (a layout change on the way)."""
    B, S, H, D = (4 if q else 16), 1024, 8, 128
    x = torch.randn(B, S, H, D, device=DEV).bfloat16()
    out = torch.empty(B, H, S, D, device=DEV, dtype=torch.float16)

    def eager():
        out.copy_(x.permute(0, 2, 1, 3))

    def ce_fn(e):
        e.copy_(out, x.permute(0, 2, 1, 3))

    def check():
        return torch.equal(out, x.permute(0, 2, 1, 3).to(torch.float16))

    return eager, ce_fn, check, x.numel() * 4, True


def case_quantize(q):
    """FP8 per group of 128 with a page gather, into a quantized cache (the torch recipe compiled)."""
    blocks, tok, hd, n, g = 256, 64, 1024, (32 if q else 128), 128
    kv = torch.randn(blocks, tok, hd, device=DEV).bfloat16()
    qc = torch.zeros(blocks, tok, hd, device=DEV, dtype=torch.float8_e4m3fn)
    sc = torch.zeros(blocks, tok, hd // g, device=DEV)
    pages = torch.randperm(blocks, device=DEV)[:n]
    slots = torch.randperm(blocks, device=DEV)[:n]

    def eager():
        _quant.native_quantize(kv, qc, sc, None, "fp8_e4m3", g, _quant.EPS, 1.0, pages, slots)

    def ce_fn(e):
        e.quantize(kv, qc, sc, fmt="fp8_e4m3", group=g, x_index=pages, index=slots)

    def check():  # the recipe's bits (compiled torch may round differently: reported, not required)
        codes, s, _ = _quant.quantize(kv[pages], g, "fp8_e4m3")
        got = qc.view(torch.uint8)[slots].reshape(-1)
        nan = got == 0x7F  # NaN codes may differ in sign between devices: positions only
        return torch.equal(got[~nan], codes[~nan]) and torch.equal(sc[slots].reshape(-1), s)

    return eager, ce_fn, check, n * tok * hd * 3, True


CASES = {"offload": case_offload, "onload": case_onload, "many_small": case_many_small,
         "transpose_cast": case_transpose_cast, "quantize": case_quantize}


def run(name, quick, reps):
    eager, ce_fn, check, nbytes, graphs_ok = CASES[name](quick)
    row = {"case": name, "MB": round(nbytes / 1e6, 1)}
    row["eager"] = timeit(eager, reps)
    modes = [("compile", None)] + ([("compile_cg", "reduce-overhead")] if graphs_ok else [])
    for label, mode in modes:
        try:
            c = compiled(eager, mode)
            warm(c, 5)
            row[label] = timeit(c, reps)
            row[label + "_ok"] = bool(check())
        except Exception as ex:  # report, keep going
            row[label] = None
            row[label + "_error"] = f"{type(ex).__name__}: {str(ex).splitlines()[0][:120]}"
    e = ce.Engine(tune_trials=3, tune_topk=4)
    warm(lambda: ce_fn(e), 80)  # explored, A/B tested and locked, as after a model's warmup
    row["copyengine"] = timeit(lambda: ce_fn(e), reps)
    row["copyengine_ok"] = bool(check())
    # copyengine inside the compiled function (0.5: never worse than torch.compile alone)
    try:
        e2 = ce.Engine(tune_trials=3, tune_topk=4)
        cc = compiled(lambda: ce_fn(e2))
        warm(cc, 80)
        row["compile_ce"] = timeit(cc, reps)
        row["compile_ce_ok"] = bool(check())
    except Exception as ex:
        row["compile_ce"] = None
        row["compile_ce_error"] = f"{type(ex).__name__}: {str(ex).splitlines()[0][:120]}"
    row["strategies"] = sorted({d["strategy"] for d in e.stats()["last"]["decisions"]})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("needs a CUDA GPU")
    reps = 10 if a.quick else 30
    rows = []
    print(f"torch {torch.__version__}, copyengine {ce.__version__}; median us per iteration")
    hdr = (f"{'case':15s} {'MB':>7s} {'eager':>9s} {'compile':>9s} {'compile_cg':>10s} {'copyengine':>10s} "
           f"{'compile+ce':>10s}  ce strategies")
    print(hdr)
    for name in a.only or CASES:
        r = run(name, a.quick, reps)
        rows.append(r)
        f = lambda k: "-" if r.get(k) is None else f"{r[k]:.0f}"
        bad = [k for k in r if k.endswith("_ok") and not r[k]] + [k for k in r if k.endswith("_error")]
        print(f"{name:15s} {r['MB']:7.1f} {f('eager'):>9s} {f('compile'):>9s} {f('compile_cg'):>10s} "
              f"{f('copyengine'):>10s} {f('compile_ce'):>10s}  {','.join(r['strategies'])}" + (f"  !! {bad}" if bad else ""))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"torch": torch.__version__, "copyengine": ce.__version__, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
