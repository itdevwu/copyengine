"""The explicit API inside torch.compile regions (docs/design/v0.5.md).

While Dynamo traces, the API emits copyengine's torch custom ops directly
(`torch.ops.copyengine.*`, which carry meta kernels and mutation
annotations), so a call is one node of the graph instead of a graph break.
The engine is passed as an integer: an Engine's handle, or 0 for the
process-wide engine the module-level functions use. Lazy views are recorded
as plain Python (`TracedLazy`) and lowered to a copy, a gather or a scatter;
a form no op takes breaks the graph explicitly and runs eagerly, with the same
results.
"""

from __future__ import annotations

import os
import statistics
import threading
import time

from . import _torch_ops

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None


def compiling() -> bool:
    return torch is not None and torch.compiler.is_compiling()


# ---- the baseline inside a compiled region: what torch.compile makes of the same op ---------
#
# The first time a call of a given kind and layout is traced, a worker thread
# (Dynamo traces only the calling thread) builds real tensors of that layout,
# compiles the op's torch equivalent with Inductor, checks its bits
# against copyengine's (NaN for NaN), and times both, copyengine explored and
# locked first. The traced call then emits the torch equivalent (Inductor's
# code, which may also fuse with its neighbours) unless copyengine is clearly
# faster: ties keep the baseline. Decisions are kept per process.
# CE_COMPILE_BASELINE=0 skips the measurement (always copyengine).

_DECISIONS: dict = {}
_REASONS: dict = {}  # per decision: why (diagnostics)
_LOCK = threading.Lock()
_MARGIN = 0.95  # copyengine must be at least 5% faster


def _meta(t):
    """A tensor's layout as constants (None: not a tensor, or symbolic sizes)."""
    if not isinstance(t, torch.Tensor):
        return None
    shape, stride = tuple(t.shape), tuple(t.stride())
    if not all(isinstance(v, int) for v in shape + stride):
        return False
    return (shape, stride, str(t.dtype).replace("torch.", ""), t.device.type, t.device.index or 0)


def _real(m, bound=None):
    """A real tensor with layout `m`: random values (an index: within `bound`, unique if asked)."""
    shape, stride, dtype, kind, index = m
    offset = 0  # (the traced tensor's offset is no constant for Dynamo; alignment aside, the same layout)
    dtype = getattr(torch, dtype)
    dev = torch.device(kind, index) if kind == "cuda" else torch.device("cpu")
    span = 1 + sum((n - 1) * s for n, s in zip(shape, stride) if n > 0)
    base = torch.empty(span + offset, dtype=dtype, device=dev)
    if dev.type == "cpu" and torch.cuda.is_available():
        base = base.pin_memory()
    r = base.as_strided(shape, stride, offset)
    n = r.numel()
    if bound is not None:
        b, unique = bound
        if unique and n <= b:
            r.copy_(torch.randperm(max(1, b), device=dev)[:n].reshape(shape))
        else:
            r.copy_(torch.randint(0, max(1, b), shape, device=dev))
    elif dtype.is_floating_point:
        r.copy_((torch.randn(shape, device=dev) * 3).to(dtype))
    elif dtype != torch.bool:
        r.copy_(torch.randint(0, 100, shape, device=dev).to(dtype))
    return r


def _same_bits(a, b) -> bool:
    """Every non-NaN value bit for bit, NaN for NaN."""
    if a.dtype.is_floating_point:
        an, bn = a.float().isnan(), b.float().isnan()
        if not torch.equal(an, bn):
            return False
        ints = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.contiguous().view(ints)[~an], b.contiguous().view(ints)[~bn])
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _kind(kind, extra):
    """(torch equivalent, copyengine op taking an engine handle first, mutated args, index bounds)."""
    ops = torch.ops.copyengine
    if kind == "copy":
        return (lambda d, s: d.copy_(s)), (lambda h, d, s: ops.copy_(d, s, h)), [0], {}
    if kind == "copy_many":
        n = extra[0]

        def native(*a):
            for d, s in zip(a[:n], a[n:]):
                d.copy_(s)
        return native, (lambda h, *a: ops.copy_many_(list(a[:n]), list(a[n:]), h)), list(range(n)), {}
    if kind == "gather":
        dim, bound = extra
        return ((lambda o, s, i: o.copy_(s.index_select(dim, i))), (lambda h, o, s, i: ops.gather_(o, s, dim, i, h)),
                [0], {2: (bound, False)})
    if kind == "scatter":
        dim, bound = extra
        return ((lambda d, i, s: d.index_copy_(dim, i, s.to(d.dtype))),
                (lambda h, d, i, s: ops.scatter_(d, i, s, dim, h)), [0], {1: (bound, True)})
    from . import _quant  # quantize / dequantize: the recipe is the baseline
    fmt, g, bh, eps, gs, static, has_z, xb, cb = extra
    code, blk = _quant.FMT[fmt], (None if bh == 1 else (bh, g))
    bounds = {}
    if xb is not None:
        bounds[4] = (xb, kind == "dequantize")
    if cb is not None:
        bounds[5] = (cb, kind == "quantize")
    if kind == "quantize":
        mutated = [1] + ([] if static else [2]) + ([3] if has_z and not static else [])
        if static:
            def native(x, q, s, z, xi, ci):
                _quant.native_quantize_static(x, q, s, z, fmt, g, gs, xi, ci, blk)
        else:
            def native(x, q, s, z, xi, ci):
                _quant.native_quantize(x, q, s, z, fmt, g, eps, gs, xi, ci, blk)
        return (native, (lambda h, x, q, s, z, xi, ci: ops.quantize_(x, q, s, z, code, g, bh, eps, gs, static, xi, ci,
                                                                     h)), mutated, bounds)

    def native(x, q, s, z, xi, ci):
        _quant.native_dequantize(q, s, z, x, fmt, g, gs, xi, ci, blk)
    return native, (lambda h, x, q, s, z, xi, ci: ops.dequantize_(q, s, x, z, code, g, bh, gs, xi, ci, h)), [0], bounds


def _measure(kind, metas, extra, reps=25):
    """'torch' (the compiled baseline) or 'ce', measured as described above."""
    from ._api import Engine
    native, with_ce, mutated, bounds = _kind(kind, extra)
    sync = torch.cuda.synchronize
    ins = [None if m is None else _real(m, bounds.get(i)) for i, m in enumerate(metas)]
    nat = [a.clone() if i in mutated else a for i, a in enumerate(ins)]
    cea = [a.clone() if i in mutated else a for i, a in enumerate(ins)]
    # Inductor directly (traced by make_fx): torch.compile would wait for Dynamo's
    # compile lock, which the tracing thread holds while it waits for this one.
    from torch.fx.experimental.proxy_tensor import make_fx

    def returning(*a):
        native(*a)
        return [a[i] for i in mutated]

    compiled = torch._inductor.compile(make_fx(returning)(*nat), nat)
    eng = Engine()
    for _ in range(40):  # copyengine explores and locks, as after a model's warmup
        with_ce(eng._handle, *cea)
    for _ in range(3):
        compiled(*nat)
    sync()
    if not all(_same_bits(nat[i], cea[i]) for i in mutated):
        return "ce", "the compiled baseline computes other bits"
    tn, tc = [], []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        compiled(*nat)
        sync()
        t1 = time.perf_counter()
        with_ce(eng._handle, *cea)
        sync()
        t2 = time.perf_counter()
        tn.append(t1 - t0)
        tc.append(t2 - t1)
    mc, mn = statistics.median(tc), statistics.median(tn)
    why = f"copyengine {mc * 1e6:.0f} us, compiled baseline {mn * 1e6:.0f} us"
    return ("ce" if mc < _MARGIN * mn else "torch"), why


def _decide(kind, metas, extra):
    if os.environ.get("CE_COMPILE_BASELINE", "1") in ("0", "off", "false") or not torch.cuda.is_available():
        return "ce"
    if any(m is False for m in metas) or not any(m is not None and m[3] == "cuda" for m in metas):
        return "ce"  # symbolic sizes, or no GPU tensor
    key = (kind, metas, extra)
    with _LOCK:
        if key in _DECISIONS:
            return _DECISIONS[key]
    out = {}

    def work():
        try:
            out["d"], out["why"] = _measure(kind, metas, extra)
        except Exception as e:  # a baseline that does not compile or run: copyengine
            out["d"], out["why"] = "ce", f"no compiled baseline: {type(e).__name__}: {str(e).splitlines()[0][:200]}"

    t = threading.Thread(target=work, name="copyengine-baseline")
    t.start()
    t.join()
    d = out.get("d", "ce")
    with _LOCK:
        _DECISIONS[key] = d
        _REASONS[key] = out.get("why", "")
    return d


if torch is not None:  # run while Dynamo traces, its result a constant of the graph
    _decide = torch._dynamo.assume_constant_result(_decide)


def available() -> bool:
    return _torch_ops.copy_ is not None and hasattr(torch.ops.copyengine, "quantize_")


def decide(kind, tensors, extra=()):
    """While tracing: 'torch' to emit the torch equivalent (torch.compile's code
    is as fast), else 'ce'."""
    return _decide(kind, tuple(_meta(t) for t in tensors), tuple(extra))


class TracedLazy:
    """`ce.lazy(t)` while Dynamo traces: the tensor and the ops, recorded."""

    def __init__(self, t, ops=(), shape=None):
        self.t, self.ops = t, ops
        self.shape = tuple(t.shape) if shape is None else shape

    def _with(self, *op):
        return TracedLazy(self.t, self.ops + (op,), _shape_after(self.shape, op))

    def permute(self, *dims):
        dims = dims[0] if len(dims) == 1 and isinstance(dims[0], (list, tuple)) else dims
        return self._with("permute", tuple(dims))

    def transpose(self, a, b):
        return self._with("transpose", a, b)

    def index_select(self, dim, index):
        return self._with("index_select", dim, index)

    def slice(self, dim, start=None, stop=None, step=1):
        return self._with("slice", dim, start, stop, step)

    def select(self, dim, index):
        return self._with("select", dim, index)

    def flip(self, dim):
        return self._with("flip", dim)

    def unsqueeze(self, dim):
        return self._with("unsqueeze", dim)

    def expand(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
        return self._with("expand", tuple(shape))

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
        return self._with("reshape", tuple(shape))

    def eager(self):
        """The same view as a `copyengine.Lazy` (after a graph break)."""
        from ._tensor import lazy
        lz = lazy(self.t)
        for op in self.ops:
            lz = getattr(lz, op[0])(*op[1:])
        return lz


def _shape_after(shape, op):
    """The shape a recorded op gives (einops asks for it at each step)."""
    s, name = list(shape), op[0]
    if name == "permute":
        return tuple(s[d] for d in op[1])
    if name == "transpose":
        a, b = op[1] % len(s), op[2] % len(s)
        s[a], s[b] = s[b], s[a]
        return tuple(s)
    if name == "index_select":
        s[op[1]] = len(op[2])
        return tuple(s)
    if name == "slice":
        _, dim, start, stop, step = op
        s[dim] = len(range(*slice(start, stop, step).indices(s[dim])))
        return tuple(s)
    if name == "select":
        del s[op[1]]
        return tuple(s)
    if name == "unsqueeze":
        d = op[1] if op[1] >= 0 else op[1] + len(s) + 1
        s.insert(d, 1)
        return tuple(s)
    if name == "expand":
        return tuple(o if n == -1 else n for o, n in zip([1] * (len(op[1]) - len(s)) + s, op[1]))
    if name == "reshape":
        new = list(op[1])
        if -1 in new:
            known = 1
            for n in new:
                if n != -1:
                    known *= n
            total = 1
            for n in s:
                total *= n
            new[new.index(-1)] = total // known
        return tuple(new)
    return tuple(s)  # flip


def _view(t, op):
    """A recorded op applied as a torch view; None where torch has no view for it."""
    name = op[0]
    if name == "permute":
        return t.permute(op[1])
    if name == "transpose":
        return t.transpose(op[1], op[2])
    if name == "slice":
        _, dim, start, stop, step = op
        idx = [slice(None)] * t.dim()
        idx[dim] = slice(start, stop, step)
        return t[tuple(idx)]
    if name == "select":
        return t.select(op[1], op[2])
    if name == "unsqueeze":
        return t.unsqueeze(op[1])
    if name == "expand":
        return t.expand(op[1])
    if name == "reshape":
        v = t.view(op[1]) if t.is_contiguous() else None
        return v
    return None


def _split(lz):
    """(base view before the index, (dim, index) or None, permutation after it) for the
    forms the ops take: views, at most one index_select with a CUDA int64 vector,
    then only permutes. None for anything else."""
    t, ops = lz.t, lz.ops
    k = 0
    while k < len(ops) and ops[k][0] != "index_select":
        t = _view(t, ops[k])
        if t is None:
            return None
        k += 1
    if k == len(ops):
        return t, None, None
    dim, index = ops[k][1], ops[k][2]
    if not (isinstance(index, torch.Tensor) and index.is_cuda and index.dtype == torch.int64 and index.dim() == 1):
        return None
    perm = list(range(t.dim()))
    for op in ops[k + 1:]:
        if op[0] == "permute":
            perm = [perm[d] for d in op[1]]
        elif op[0] == "transpose":
            perm[op[1]], perm[op[2]] = perm[op[2]], perm[op[1]]
        else:
            return None
    return t, (dim if dim >= 0 else dim + t.dim(), index.contiguous()), perm


def _inverse(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def copy_(engine: int, dst, src, fn=None) -> bool:
    """Emits the op for `dst = fn(src)`; False: no op takes it (the caller breaks the graph)."""
    ops = torch.ops.copyengine
    identity = fn is None or (fn.scale == 1.0 and fn.bias == 0.0 and not fn.round and fn.clamp is None)
    s = _split(src) if isinstance(src, TracedLazy) else (src, None, None)
    d = _split(dst) if isinstance(dst, TracedLazy) else (dst, None, None)
    if s is None or d is None:
        return False
    (st, sidx, sperm), (dt, didx, dperm) = s, d
    if sidx is not None and didx is not None:
        return False
    if sidx is not None:  # dst = base.index_select(dim, i).permute(p): gather into dst.permute(p^-1)
        if not identity:
            return False
        gather(engine, st, sidx[1], sidx[0], dt.permute(_inverse(sperm)))
        return True
    if didx is not None:  # base.index_select(dim, i).permute(p) = src: scatter src.permute(p^-1)
        if not identity:
            return False
        scatter_(engine, dt, didx[1], st.permute(_inverse(dperm)), didx[0])
        return True
    if identity:
        if decide("copy", [dt, st]) == "torch":
            dt.copy_(st)
        else:
            ops.copy_(dt, st, engine)
    else:
        lo, hi = fn.clamp or (0.0, 0.0)
        ops.copy_fn_(dt, st, engine, fn.scale, fn.bias, fn.round, fn.clamp is not None, lo, hi)
    return True


def copy_many(engine: int, dsts, srcs) -> bool:
    dsts, srcs = list(dsts), list(srcs)
    if not all(isinstance(t, torch.Tensor) for t in dsts + srcs):
        return False
    n = len(dsts)

    def native(*a):
        for d, s in zip(a[:n], a[n:]):
            d.copy_(s)

    if decide("copy_many", dsts + srcs, (n,)) == "torch":
        native(*dsts, *srcs)
    else:
        torch.ops.copyengine.copy_many_(dsts, srcs, engine)
    return True


def gather(engine: int, src, index, dim, out) -> bool:
    if not (isinstance(src, torch.Tensor) and isinstance(index, torch.Tensor) and index.is_cuda):
        return False
    idx = index.to(torch.int64).contiguous()
    dim = dim if dim >= 0 else dim + src.dim()
    if decide("gather", [out, src, idx], (dim, src.shape[dim])) == "torch":
        out.copy_(src.index_select(dim, idx))
    else:
        torch.ops.copyengine.gather_(out, src, dim, idx, engine)
    return True


def scatter_(engine: int, dst, index, src, dim) -> bool:
    if not (isinstance(dst, torch.Tensor) and isinstance(index, torch.Tensor) and index.is_cuda):
        return False
    idx = index.to(torch.int64).contiguous()
    dim = dim if dim >= 0 else dim + dst.dim()
    same = isinstance(src, torch.Tensor) and src.dim() == dst.dim() and src.device == dst.device
    if same and decide("scatter", [dst, idx, src], (dim, dst.shape[dim])) == "torch":
        dst.index_copy_(dim, idx, src.to(dst.dtype))
    else:
        torch.ops.copyengine.scatter_(dst, idx, src, dim, engine)
    return True


def _index(ix):
    if ix is None:
        return None
    if isinstance(ix, torch.Tensor):
        return ix.to(torch.int64).contiguous()
    return None  # lists / arrays: not traced (the caller breaks the graph)


def tile(engine: int, quantize: bool, x, q, scale, zero, fmt, group, eps, gs, static, x_index, index,
         block=None) -> bool:
    from . import _quant
    if fmt not in _quant.FMT:
        raise ValueError(f"unknown format {fmt!r}; one of {sorted(_quant.FMT)}")
    xi, ci = _index(x_index), _index(index)
    if (x_index is not None and xi is None) or (index is not None and ci is None):
        return False
    D = x.shape[-1]
    bh = 1
    if block is not None:
        if fmt not in _quant.BOX_FORMATS:
            raise ValueError(f"block= is for {', '.join(_quant.BOX_FORMATS)}")
        bh, group = int(block[0]), int(block[1])
    g = _quant.UNIT.get(fmt, group if group is not None else D)
    if group is not None and fmt in _quant.UNIT and group != g:
        raise ValueError(f"{fmt} uses units of {g} values")
    if fmt == "uint4" and zero is None:
        raise ValueError("uint4 needs a zero-point tensor (`zero`)")
    z = zero if fmt == "uint4" else None
    code, blk = _quant.FMT[fmt], (None if block is None else (bh, g))
    args = [x, q, scale, z, xi, ci]
    extra = (fmt, g, bh, float(eps), float(gs), bool(static), z is not None,
             None if xi is None else x.shape[0], None if ci is None else q.shape[0])
    native = _kind("quantize" if quantize else "dequantize", extra)[0]
    if all(a is None or a.device.type == "cuda" for a in args) and decide(
            "quantize" if quantize else "dequantize", args, extra) == "torch":
        native(*args)
    elif quantize:
        torch.ops.copyengine.quantize_(x, q, scale, z, code, g, bh, float(eps), float(gs), bool(static), xi, ci, engine)
    else:
        torch.ops.copyengine.dequantize_(q, scale, x, z, code, g, bh, float(gs), xi, ci, engine)
    return True
