"""Validation kit: one command on a machine we cannot log into, one JSON report back.

    git clone --branch <tag> https://github.com/itdevwu/copyengine && cd copyengine
    pip install --pre copyengine==<version> pytest  # the release under test (pytest: peer tests)
    python tests/fuzz/kit.py --out report.json  # add --bench for the benchmarks

The report holds the environment (GPU generations, links between GPUs,
copyengine's admission verdicts per GPU and per ordered pair of GPUs), with
two GPUs or more tests/python/test_peer.py's results, the fuzzer's results (one process per
seed, every mode, every strategy forced; each failure with its seed and mode so
it reproduces with `python tests/fuzz/fuzz.py <seed> 1 <mode>`) and, with
--bench, the benchmark outputs. It is rewritten after every seed: a crash or a
lost session keeps what ran. It holds machine data: do not commit it.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODES = "forced,explicit,scope,freeze,plan,graph,threads"


def environment():
    import torch
    import copyengine as ce

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "copyengine": ce.__version__,
        "platform": platform.platform(),
        "gpus": [],
        "peer_access": {},
        "admission": [],
    }
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        env["gpus"].append({"index": i, "name": p.name, "sm": f"sm_{p.major}{p.minor}", "sms": p.multi_processor_count,
                            "memory_gb": round(p.total_memory / 2**30, 1)})
    for a in range(n):
        for b in range(n):
            if a != b:
                env["peer_access"][f"{a}->{b}"] = bool(torch.cuda.can_device_access_peer(a, b))
    for i in range(n):  # one copy per GPU: admission runs, its verdicts are in stats()["cuda"]
        e = ce.Engine()
        x = torch.ones(1024, device=f"cuda:{i}")
        e.copy_(torch.empty_like(x), x)
        torch.cuda.synchronize(i)
        cuda = e.stats().get("cuda", [])
        env["admission"].append(next((c for c in cuda if c.get("index", c.get("device")) == i), cuda[i] if i < len(cuda) else {}))
    if n > 1:  # every ordered pair: a copy between them admits it; its verdicts are in stats()["cuda"][a]["peers"]
        e = ce.Engine()
        for a in range(n):
            for b in range(n):
                if a != b:
                    try:
                        e.copy_(torch.empty(1024, device=f"cuda:{b}"), torch.ones(1024, device=f"cuda:{a}"))
                        torch.cuda.synchronize(b)
                    except Exception as ex:
                        env["peer_access"][f"{a}->{b} copy"] = f"{type(ex).__name__}: {ex}"
        env["peers"] = {str(c.get("index")): c.get("peers", {}) for c in e.stats().get("cuda", [])}
    try:
        from copyengine import _core
        pinned = torch.zeros(16).pin_memory()
        env["host_memory"] = {"pin_memory": _core.host_memory(pinned.data_ptr())}
        import ctypes
        buf = (ctypes.c_ubyte * 4096)()
        rt = torch.cuda.cudart()
        if rt.cudaHostRegister(ctypes.addressof(buf), 4096, 1) == rt.cudaError.success:
            env["host_memory"]["cudaHostRegister"] = _core.host_memory(ctypes.addressof(buf))
            rt.cudaHostUnregister(ctypes.addressof(buf))
    except Exception as ex:  # informational only
        env["host_memory"] = {"error": f"{type(ex).__name__}: {ex}"}
    return env


def fuzz_seed(seed, modes, timeout):
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, os.path.join(HERE, "fuzz.py"), str(seed), "1", modes],
                           capture_output=True, text=True, timeout=timeout)
        out = p.stdout + p.stderr
        code = p.returncode
    except subprocess.TimeoutExpired as ex:
        out = (ex.stdout or "") + (ex.stderr or "") if isinstance(ex.stdout, str) else ""
        code = "timeout"
    fails = [line for line in out.splitlines() if line.startswith("FAIL")]
    errors = [line for line in out.splitlines() if "Error" in line and not line.startswith(" ")][-3:]
    return {"seed": seed, "seconds": round(time.time() - t0, 1), "exit": code, "fails": fails,
            "errors": errors if (fails or code not in (0,)) else []}


def peer_tests(timeout):
    """tests/python/test_peer.py (two GPUs or more), when pytest is installed: with torch's
    default allocator and with expandable segments (memory peers reach only once torch maps it)."""
    try:
        import pytest  # noqa: F401
    except ImportError:
        return {"skipped": "pytest is not installed (pip install pytest)"}
    path = os.path.join(ROOT, "tests", "python", "test_peer.py")
    out = {}
    for name, conf in (("default", None), ("expandable_segments", "expandable_segments:True")):
        env = dict(os.environ)
        if conf:
            env["PYTORCH_CUDA_ALLOC_CONF"] = conf
        try:
            p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", path], capture_output=True, text=True,
                               timeout=timeout, cwd=ROOT, env=env)
            lines = (p.stdout + p.stderr).splitlines()
            out[name] = {"exit": p.returncode, "summary": lines[-1] if lines else "",
                         "failures": [x for x in lines if x.startswith(("FAILED", "ERROR", "E  "))][:40]}
        except subprocess.TimeoutExpired:
            out[name] = {"exit": "timeout"}
    return out


def bench(script, timeout):
    path = os.path.join(ROOT, "bench", script)
    if not os.path.exists(path):
        return {"error": f"{script} not found"}
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout, cwd=ROOT)
        return {"exit": p.returncode, "stdout": p.stdout[-20000:], "stderr": p.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        return {"exit": "timeout"}


def main():
    ap = argparse.ArgumentParser(description="copyengine validation kit")
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--modes", default=MODES)
    ap.add_argument("--timeout", type=int, default=600, help="seconds per seed")
    ap.add_argument("--bench", action="store_true", help="also run bench/bench_torch.py, bench/bench_compile.py and (two GPUs) bench/bench_p2p.py")
    args = ap.parse_args()

    report = {"kit": 1, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args)}

    def save():
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)

    report["environment"] = environment()
    save()
    if len(report["environment"]["gpus"]) > 1:
        report["peer_tests"] = peer_tests(1800)
        save()
        for k, v in report["peer_tests"].items():
            print(f"peer tests ({k}):", v.get("summary", v) if isinstance(v, dict) else v, flush=True)
    report["fuzz"] = {"modes": args.modes, "seeds": [], "failing_seeds": []}
    for seed in range(args.first, args.first + args.seeds):
        r = fuzz_seed(seed, args.modes, args.timeout)
        report["fuzz"]["seeds"].append(r)
        if r["fails"] or r["exit"] != 0:
            report["fuzz"]["failing_seeds"].append(seed)
            print(f"seed {seed}: " + "; ".join(r["fails"] + r["errors"])[:300], flush=True)
        save()
    if args.bench:
        report["bench"] = {"bench_torch": bench("bench_torch.py", 3600), "bench_compile": bench("bench_compile.py", 3600)}
        if len(report["environment"]["gpus"]) > 1:
            report["bench"]["bench_p2p"] = bench("bench_p2p.py", 3600)
        save()
    f = report["fuzz"]["failing_seeds"]
    print(f"{len(report['fuzz']['seeds'])} seeds, {len(f)} failing: {f[:20]}; report: {args.out}", flush=True)


if __name__ == "__main__":
    main()
