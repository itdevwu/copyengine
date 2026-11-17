"""Random-program fuzzer: copyengine vs torch executing the same copies in order.

A program is a sequence of copies between views of a few base tensors (CUDA,
pinned and pageable host memory) and temporaries. Views are permute + per-dim
slice or index_select (host numpy or CUDA int64 indices; unique on
destinations). Copies may overlap one another (RAW / WAR / WAW across ops),
never within one op. Each program runs in several modes, each many times (so
exploration tries different strategies), and every run is compared bit for bit
with torch running the copies one after another.

Modes:
  explicit  ce.copy_ of lazy views, all in one scope (one planned flush)
  scope     plain torch dst.copy_(src) inside eng.scope() (interception); index-free ops only
  plan      the program as a step plan (temporaries allocated inside); new index values per call
  freeze    one explicit flush, frozen, replayed

Also `forced`: every strategy (Engine(_force=...)) wherever the executor offers it, the most
conservative one elsewhere, so a failure belongs to the forced strategy. FUZZ_SKIP=a,b leaves
strategies out.

usage: python fuzz.py [first_seed] [count] [modes]   (`just fuzz` runs one process per seed:
a CUDA fault poisons the process)
"""

import sys
import traceback

import numpy as np
import torch

import copyengine as ce

DEV = "cuda"
# Host memory kinds: pinned (cudaHostAlloc, mapped), registered (cudaHostRegister: pinned, maybe
# not mapped at its own address), pageable. With several GPUs, CUDA bases spread over them.
KINDS = ("cuda", "pinned", "registered", "pageable")
GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
_REGISTERED = []  # (ctypes buffer) kept alive and registered for the process
STRATEGIES = ("native", "passthrough", "runs", "runs_parallel", "batch", "kernel", "staged", "bounce", "fused",
              "lowered", "bulk", "tma2d")
import os as _os  # noqa: E402
STRATEGIES = tuple(s for s in STRATEGIES if s not in _os.environ.get("FUZZ_SKIP", "").split(","))


def registered_like(t):
    """A copy of host tensor t in memory registered with cudaHostRegister (Portable): pages of
    its own (mmap), so no registration shares a page with another and none fails for that."""
    import ctypes
    import mmap
    n = t.numel() * t.element_size()
    buf = mmap.mmap(-1, max(mmap.PAGESIZE, -(-n // mmap.PAGESIZE) * mmap.PAGESIZE))
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    rt = torch.cuda.cudart()
    err = rt.cudaHostRegister(addr, len(buf), 1)
    if err != rt.cudaError.success:  # (rarely, a resource limit): this base is pinned instead, said so
        print(f"WARN fuzz: cudaHostRegister failed ({err}): a registered base is pinned memory instead",
              file=sys.stderr, flush=True)
        return t.pin_memory()
    _REGISTERED.append((buf, addr))
    r = torch.frombuffer(buf, dtype=torch.uint8)[:n].view(t.dtype).view(t.shape)
    r.copy_(t)
    return r


def make_base(rng, kind, dtype):
    shape = tuple(int(rng.integers(4, 11)) for _ in range(3))
    t = torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).to(dtype)
    if kind == "cuda":
        return t.to(f"cuda:{int(rng.integers(0, GPUS))}" if GPUS > 1 else DEV)
    if kind == "registered":
        return registered_like(t)
    return t.pin_memory() if kind == "pinned" else t


def kind_of(t):
    """The memory kind a base was made with (copies of it are made alike)."""
    if t.is_cuda:
        return "cuda"
    if not t.is_pinned():
        return "pageable"
    return "registered" if any(t.data_ptr() == addr for _, addr in _REGISTERED) else "pinned"


def copy_like(t, kind):
    if kind == "registered":
        return registered_like(t)
    if kind == "pinned":
        return t.clone().pin_memory()
    return t.clone()


class View:
    """base.permute(perm), then per dimension of the permuted tensor a slice or an index list."""

    def __init__(self, base, perm, sel):
        self.base, self.perm, self.sel = base, perm, sel  # sel[d] = ("s", a, b) or ("i", np.ndarray)

    def shape(self, bases):
        return tuple((s[2] - s[1]) if s[0] == "s" else len(s[1]) for s in self.sel)

    def indexed(self):
        return [d for d, s in enumerate(self.sel) if s[0] == "i"]

    def offsets(self, bases):
        """Storage offsets (elements) of every element, for overlap checks."""
        b = bases[self.base]
        probe = torch.arange(b.numel()).view(b.shape)
        return self.torch_read(probe).reshape(-1)

    def torch_read(self, t):
        x = t.permute(*self.perm)
        for d, s in enumerate(self.sel):
            if s[0] == "s":
                x = x.narrow(d, s[1], s[2] - s[1])
            else:
                x = x.index_select(d, torch.as_tensor(s[1], device=x.device))
        return x

    def torch_write(self, t, value):
        """t's view := value, as torch would (index lists by index assignment)."""
        x = t.permute(*self.perm)
        idx = []
        for d, s in enumerate(self.sel):
            if s[0] == "s":
                idx.append(torch.arange(s[1], s[2], device=x.device))
            else:
                idx.append(torch.as_tensor(s[1], device=x.device))
        grids = torch.meshgrid(*idx, indexing="ij")
        x[grids] = value.to(x.dtype).to(x.device)

    def lazy(self, t, cuda_index, cache=None):
        x = ce.lazy(t).permute(*self.perm)
        for d, s in enumerate(self.sel):
            if s[0] == "s":
                x = x.slice(d, s[1], s[2])
            else:
                idx = s[1]
                if cuda_index and t.is_cuda:
                    key = (id(s[1]), t.device)
                    if cache is not None and key in cache:
                        idx = cache[key]
                    else:  # (made before a capture: no allocation while capturing)
                        idx = torch.as_tensor(idx, device=t.device)
                        if cache is not None:
                            cache[key] = idx
                x = x.index_select(d, idx)
        return x

    def plain(self, t):
        """A torch view (no index lists), for intercepted copies."""
        x = t.permute(*self.perm)
        for d, s in enumerate(self.sel):
            x = x.narrow(d, s[1], s[2] - s[1])
        return x


def make_view(rng, bases, bi, shape, allow_index, dst):
    b = bases[bi]
    perm = tuple(int(v) for v in rng.permutation(3))
    pshape = tuple(b.shape[p] for p in perm)
    if any(n > m for n, m in zip(shape, pshape)):
        return None
    sel = []
    for d in range(3):
        n, m = shape[d], pshape[d]
        if allow_index and rng.random() < 0.25:
            idx = rng.permutation(m)[:n] if dst else rng.integers(0, m, n)
            sel.append(("i", idx.astype(np.int64)))
        else:
            a = int(rng.integers(0, m - n + 1))
            sel.append(("s", a, a + n))
    return View(bi, perm, sel)


def make_program(seed):
    rng = np.random.default_rng(seed)
    nb = int(rng.integers(2, 5))
    dtypes = [torch.float32, torch.float32, torch.bfloat16]
    bases = [make_base(rng, KINDS[int(rng.integers(0, len(KINDS)))], dtypes[int(rng.integers(0, 3))]) for _ in range(nb)]
    if not any(b.is_cuda for b in bases):
        bases[0] = bases[0].to(DEV)
    ops = []
    tmps = 0
    for _ in range(int(rng.integers(2, 9))):
        shape = tuple(int(rng.integers(1, 5)) for _ in range(3))
        if rng.random() < 0.2:  # a temporary, written now, read by a later op
            tmps += 1
        for _ in range(20):
            si, di = int(rng.integers(0, nb)), int(rng.integers(0, nb))
            sv = make_view(rng, bases, si, shape, True, False)
            dv = make_view(rng, bases, di, shape, True, True)
            if sv is None or dv is None:
                continue
            # ce needs a GPU on one side; torch would overlap within one op otherwise
            if not (bases[si].is_cuda or bases[di].is_cuda):
                continue
            if bases[si].dtype != bases[di].dtype and not (bases[di].dtype.is_floating_point):
                continue
            if si == di and np.intersect1d(sv.offsets(bases).numpy(), dv.offsets(bases).numpy()).size:
                continue
            ops.append(("copy", dv, sv))
            break
    return bases, ops


def reference(bases, ops):
    bs = [b.clone() for b in bases]
    for kind, dv, sv in ops:
        a = sv.torch_read(bs[sv.base]).clone()  # torch's overlap check is conservative
        if kind == "swap":  # dv and sv exchange contents
            b = dv.torch_read(bs[dv.base]).clone()
            sv.torch_write(bs[sv.base], b)
        dv.torch_write(bs[dv.base], a)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return bs


def same(a, b):
    return a.dtype == b.dtype and torch.equal(a.cpu().view(torch.uint8) if a.dtype != torch.bfloat16 else a.cpu().view(torch.int16),
                                              b.cpu().view(torch.uint8) if b.dtype != torch.bfloat16 else b.cpu().view(torch.int16))


def check(tag, got, want, seed, it):
    for k, (g, w) in enumerate(zip(got, want)):
        if not same(g, w):
            return f"seed {seed} mode {tag} run {it}: base {k} differs ({(g.cpu().float() != w.cpu().float()).sum().item()} elements)"
    return None


def run_explicit(eng, bases, ops, cuda_index, cache=None):
    with eng.scope():
        for _, dv, sv in ops:
            eng.copy_(dv.lazy(bases[dv.base], cuda_index, cache), sv.lazy(bases[sv.base], cuda_index, cache))


def fuzz_one(seed, modes, runs=24):
    init, ops = make_program(seed)
    if not ops:
        return []
    want = reference(init, ops)
    errs = []

    kinds = [kind_of(b) for b in init]

    def fresh():
        return [copy_like(b, k) for b, k in zip(init, kinds)]

    if "forced" in modes:  # every strategy where the executor offers it, one run each
        for name in STRATEGIES:
            for cuda_index in (False, True):
                eng = ce.Engine(_force=name)
                bs = fresh()
                try:
                    run_explicit(eng, bs, ops, cuda_index)
                    torch.cuda.synchronize()
                except (ce.Unsupported, ce.NeedsMaterialize, ce.UnsafeCopy):
                    continue
                e = check(f"forced={name}(cuda_index={cuda_index})", bs, want, seed, 0)
                if e:
                    errs.append(e)
    if "explicit" in modes:
        for cuda_index in (False, True):
            eng = ce.Engine()
            for it in range(runs):
                bs = fresh()
                try:
                    run_explicit(eng, bs, ops, cuda_index)
                    eng.synchronize() if hasattr(eng, "synchronize") else None
                    torch.cuda.synchronize()
                except (ce.Unsupported, ce.NeedsMaterialize, ce.UnsafeCopy):
                    break
                e = check(f"explicit(cuda_index={cuda_index})", bs, want, seed, it)
                if e:
                    errs.append(e)
                    break
    if "scope" in modes:
        plain = [op for op in ops if not op[1].indexed() and not op[2].indexed()]
        if plain:
            want_p = reference(init, plain)
            eng = ce.Engine()
            for it in range(runs):
                bs = fresh()
                with eng.scope():
                    for _, dv, sv in plain:
                        dv.plain(bs[dv.base]).copy_(sv.plain(bs[sv.base]), non_blocking=True)
                torch.cuda.synchronize()
                e = check("scope", bs, want_p, seed, it)
                if e:
                    errs.append(e)
                    break
    if "freeze" in modes:
        eng = ce.Engine()
        try:
            for _ in range(30):  # settle, then freeze the last flush
                bs = fresh()
                run_explicit(eng, bs, ops, True)
            torch.cuda.synchronize()
            fr = eng.freeze()
            for it in range(6):
                for b, b0 in zip(bs, init):
                    b.copy_(b0)
                torch.cuda.synchronize()
                fr.run()
                torch.cuda.synchronize()
                e = check("freeze", bs, want, seed, it)
                if e:
                    errs.append(e)
                    break
        except (ce.Unsupported, ce.NeedsMaterialize, ce.UnsafeCopy):
            pass
    if "plan" in modes:
        errs += fuzz_plan(seed, init, ops, fresh)
    if "graph" in modes:  # the program captured in a CUDA graph after warmup, replayed
        eng = ce.Engine()
        bs = fresh()
        cache = {}
        try:
            for _ in range(30):
                run_explicit(eng, bs, ops, True, cache)
            torch.cuda.synchronize()
            g, side = torch.cuda.CUDAGraph(), torch.cuda.Stream()
            with torch.cuda.graph(g, stream=side):
                run_explicit(eng, bs, ops, True, cache)
            for it in range(3):
                for b, b0 in zip(bs, init):
                    b.copy_(b0)
                torch.cuda.synchronize()
                g.replay()
                torch.cuda.synchronize()
                e = check("graph", bs, want, seed, it)
                if e:
                    errs.append(e)
                    break
        except (ce.Unsupported, ce.NeedsMaterialize, ce.UnsafeCopy, RuntimeError) as ex:
            if not any(w in str(ex) for w in ("captur", "graph")) and \
                    not isinstance(ex, (ce.Unsupported, ce.NeedsMaterialize, ce.UnsafeCopy)):
                errs.append(f"seed {seed} mode graph: {type(ex).__name__}: {str(ex).splitlines()[0][:160]}")
    return errs


def fuzz_plan(seed, init, ops, fresh, calls=90):
    """The program as a step plan: bases as arguments, index lists as (host) arguments whose
    values change every call, and a temporary round trip: some ops go through a tensor the step
    allocates and drops (dst -> tmp -> dst2)."""
    rng = np.random.default_rng(seed + 10**6)
    errs = []
    eng = ce.Engine()

    # Index-free ops of one dtype, in a swappable layout, become swaps through a temporary
    # (tmp = s; s = d; d = tmp): the step-plan composition must not read s after it changed.
    swaps = {j for j, (_, dv, sv) in enumerate(ops) if not dv.indexed() and not sv.indexed()
             and init[dv.base].dtype == init[sv.base].dtype and rng.random() < 0.4}

    def step(bases, *idx):
        k = 0
        with eng.scope():
            for j, (_, dv, sv) in enumerate(ops):
                sel_d, sel_s = [], []
                for v, out in ((dv, sel_d), (sv, sel_s)):
                    for s in v.sel:
                        if s[0] == "i":
                            out.append(idx[k]); k += 1
                v_d = _with_idx(dv, sel_d).lazy(bases[dv.base], False)
                v_s = _with_idx(sv, sel_s).lazy(bases[sv.base], False)
                if j in swaps:
                    tmp = torch.empty(sv.shape(bases), dtype=bases[sv.base].dtype, device=DEV)
                    eng.copy_(tmp, v_s)
                    eng.copy_(v_s, v_d)
                    eng.copy_(v_d, ce.lazy(tmp))
                elif dv.indexed() or sv.indexed():
                    eng.copy_(v_d, v_s)
                else:  # through a temporary the step allocates and drops
                    shape = sv.shape(bases)
                    tmp = torch.empty(shape, dtype=bases[sv.base].dtype, device=DEV)
                    eng.copy_(tmp, v_s)
                    eng.copy_(v_d, ce.lazy(tmp))

    p = eng.plan(step)
    bs = fresh()  # the same buffers every call (a step plan replays on them), contents reset
    for it in range(calls):
        # new index values, same lengths (destinations stay unique)
        cur_ops, idx = [], []
        for _, dv, sv in ops:
            nd, ns = [], []
            for v, out, dst in ((dv, nd, True), (sv, ns, False)):
                b = init[v.base]
                pshape = [b.shape[q] for q in v.perm]
                for d, s in enumerate(v.sel):
                    if s[0] == "i":
                        n, m = len(s[1]), pshape[d]
                        a = (rng.permutation(m)[:n] if dst else rng.integers(0, m, n)).astype(np.int64)
                        out.append(a)
            cur_ops.append(("swap" if len(cur_ops) in swaps else "copy", _with_idx(dv, nd), _with_idx(sv, ns)))
            idx += nd + ns
        bad = [op for op in cur_ops if op[1].base == op[2].base and
               np.intersect1d(op[1].offsets(init).numpy(), op[2].offsets(init).numpy()).size]
        if bad:
            continue
        want = reference(init, cur_ops)
        for b, b0 in zip(bs, init):
            b.copy_(b0)
        torch.cuda.synchronize()
        try:
            p(bs, *idx)
            torch.cuda.synchronize()
        except (ce.Unsupported, ce.NeedsMaterialize, ce.UnsafeCopy):
            return errs
        e = check("plan", bs, want, seed, it)
        if e:
            lay = next(iter(p._layouts.values()))
            errs.append(e + f" (traced={bool(lay['traces'])})")
            break
    return errs


def _with_idx(v, arrays):
    sel, k = [], 0
    for s in v.sel:
        if s[0] == "i":
            sel.append(("i", arrays[k])); k += 1
        else:
            sel.append(s)
    return View(v.base, v.perm, sel)


def main():
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    modes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["forced", "explicit", "scope", "freeze", "plan"]
    fails = 0
    for seed in range(first, first + count):
        try:
            errs = fuzz_one(seed, modes)
        except Exception:
            errs = [f"seed {seed}: crash\n" + traceback.format_exc(limit=4)]
        for e in errs:
            fails += 1
            print("FAIL", e, flush=True)
    print(f"done: {count} programs, {fails} failures", flush=True)


if __name__ == "__main__":
    main()
