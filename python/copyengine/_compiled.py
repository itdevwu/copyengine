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
    if _symbolic(shape) or _symbolic(stride):  # (isinstance(v, int) would ask Dynamo for a constant)
        return False
    return (shape, stride, str(t.dtype).replace("torch.", ""), t.device.type, t.device.index or 0)


def _real(m, bound=None, pinned=True):
    """A real tensor with layout `m`: random values (an index: within `bound`, unique if asked);
    host memory pinned or pageable."""
    shape, stride, dtype, kind, index = m
    offset = 0  # (the traced tensor's offset is no constant for Dynamo; alignment aside, the same layout)
    dtype = getattr(torch, dtype)
    dev = torch.device(kind, index) if kind == "cuda" else torch.device("cpu")
    span = 1 + sum((n - 1) * s for n, s in zip(shape, stride) if n > 0)
    base = torch.empty(span + offset, dtype=dtype, device=dev)
    if dev.type == "cpu" and pinned and torch.cuda.is_available():
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
        wide = torch.float64 if dtype == torch.float64 else torch.float32  # f64 samples keep f64-only values
        v = torch.randn(shape, device=dev, dtype=wide) * 3
        _edges(v.view(-1), _FLOAT_EDGES + (_F64_EDGES if wide == torch.float64 else []))
        r.copy_(v.to(dtype))
    elif dtype != torch.bool:
        v = torch.randint(0, 100, shape, device=dev).to(dtype)
        info = torch.iinfo(dtype)
        _edges(v.view(-1), [info.min, info.max, 0, -1 if info.min < 0 else 1])
        r.copy_(v)
    return r


# Values where implementations disagree most (rounding ties, saturation, NaN / inf,
# denormals, FP8 and integer ranges): part of every sample the bits are checked on.
_FLOAT_EDGES = [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 448.0, 449.0, 464.0, -448.0, 57344.0,
                61440.0, 1e30, -1e30, 3.4e38, 1e-40, -1e-40, 2.0 ** 31, -(2.0 ** 31) - 1, 0.5, 1.5, 2.5, -2.5,
                127.5, -128.5, 255.5, 1.0 + 2.0 ** -8, 1.0 + 2.0 ** -9, 1.0 + 2.0 ** -11, 1.0 + 2.0 ** -12,
                65504.0, 65520.0, 6e-8]
# Only representable in float64: beyond float32's range, below its denormals, and ties of the
# narrow types that float32 would round first (double rounding).
_F64_EDGES = [1e300, -1e300, 1e-310, 1.0 + 2.0 ** -9 + 2.0 ** -40, 1.0 + 2.0 ** -12 + 2.0 ** -40,
              3.5e38, 65519.99999999]


def _edges(flat, values):
    """Spread `values` over a flat sample (every k-th element), so every unit / row sees some."""
    n = flat.numel()
    if n == 0:
        return
    k = max(1, n // (4 * len(values)))
    for j, x in enumerate(values * 4):
        if j * k >= n:
            break
        flat[j * k] = x


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
    if kind == "copy_fn":  # dst = fn(src): the catalog's steps as torch ops (bits checked on edge values)
        scale, bias, rnd, clamp, lo, hi = extra

        def native(d, s):
            v = s.float() * scale
            v = v + bias
            if rnd:
                v = torch.round(v)
            if clamp:
                v = v.clamp(lo, hi)
            d.copy_(v)
        return native, (lambda h, d, s: ops.copy_fn_(d, s, h, scale, bias, rnd, clamp, lo, hi)), [0], {}
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
    """'torch' (the compiled baseline) or 'ce', measured as described above. Host
    tensors may be pinned or pageable at run time (the trace cannot tell): both
    are measured, and copyengine is taken only if it wins with both."""
    host = any(m is not None and m[3] == "cpu" for m in metas)
    whys = []
    for pinned in ((True, False) if host else (True,)):
        d, why = _measure_one(kind, metas, extra, reps, pinned)
        whys.append(("pinned: " if pinned and host else "pageable: " if host else "") + why)
        if d == "torch":
            return "torch", "; ".join(whys)
    return "ce", "; ".join(whys)


def _measure_one(kind, metas, extra, reps, pinned):
    from ._api import Engine
    native, with_ce, mutated, bounds = _kind(kind, extra)
    sync = torch.cuda.synchronize
    ins = [None if m is None else _real(m, bounds.get(i), pinned) for i, m in enumerate(metas)]
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


def _pairs_same_bits(kind, metas, extra) -> bool:
    """For copies, whether torch's conversion gives copyengine's bits for every (dst, src)
    dtype pair (the eager rule, same_bits): elsewhere no baseline is offered."""
    from ._tensor import same_bits_dtypes

    def ok(d, s):
        return same_bits_dtypes(getattr(torch, d[2]), getattr(torch, s[2]))
    if kind == "copy":
        return ok(metas[0], metas[1])
    if kind == "copy_fn":
        return True  # computation: checked on edge-value samples (the catalog's recipe is the reference)
    if kind == "copy_many":
        n = extra[0]
        return all(ok(d, s) for d, s in zip(metas[:n], metas[n:2 * n]))
    if kind == "gather":  # [out, src, idx]
        return ok(metas[0], metas[1])
    if kind == "scatter":  # [dst, idx, src]
        return ok(metas[0], metas[2])
    return True  # tile ops: checked on edge-value samples (their recipe is the reference)


def _decide(kind, metas, extra):
    if os.environ.get("CE_COMPILE_BASELINE", "1") in ("0", "off", "false") or not torch.cuda.is_available():
        return "ce"
    if any(m is False for m in metas) or not any(m is not None and m[3] == "cuda" for m in metas):
        return "ce"  # symbolic sizes, or no GPU tensor
    if not _pairs_same_bits(kind, metas, extra):
        return "ce"  # a dtype pair where torch's bits differ: no baseline, whatever is faster
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


def _symbolic(v) -> bool:
    """Outside Dynamo: whether `v` holds symbolic sizes. (While Dynamo traces,
    symbolic ints pass for ints: see static().)"""
    if isinstance(v, (tuple, list)):
        return any(_symbolic(x) for x in v)
    return isinstance(v, (torch.SymInt, torch.SymFloat, torch.SymBool))


# Whether a value is a constant while Dynamo traces (static sizes), asked of
# the compiler itself: traced code cannot tell a symbolic int from an int, and
# a constant-result function given one breaks the graph. comptime runs in the
# compiler at this point of the trace and leaves its answer for _static_now(),
# a constant of the graph (no guard on the answer's storage).
_STATIC = {"v": True}


def _probe_static(ctx):
    _STATIC["v"] = ctx.get_local("v").is_python_constant()


def _static_now():
    return _STATIC["v"]


def static(v) -> bool:
    """While tracing: `v` (sizes, tuples of them) has no symbolic part."""
    if torch is None or not compiling():
        return not _symbolic(v)
    from torch._dynamo.comptime import comptime
    comptime(_probe_static)
    return _static_now()


if torch is not None:
    _static_now = torch._dynamo.assume_constant_result(_static_now)


def decide(kind, tensors, extra=()):
    """While tracing: 'torch' to emit the torch equivalent (torch.compile's code
    is as fast), else 'ce'. Symbolic sizes (dynamic shapes) take 'ce' here,
    before anything symbolic reaches the constant-result function (Dynamo
    cannot make constants of them: the graph would break)."""
    if os.environ.get("CE_COMPILE_BASELINE", "1") in ("0", "off", "false"):
        return "ce"
    metas = tuple(_meta(t) for t in tensors)
    extra = tuple(extra)
    if not static((metas, extra)) or any(m is False for m in metas):
        return "ce"
    return _decide(kind, metas, extra)


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

    def narrow(self, dim, start, length):
        return self._with("slice", dim, start, start + length, 1)

    def __getitem__(self, key):
        from ._tensor import apply_index
        return apply_index(self, key)

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


def _view_stride(shape, stride, new):
    """Strides of `new` as a view of (shape, stride), or None: torch's computeStride."""
    numel = 1
    for n in shape:
        numel *= n
    if numel == 0:
        return [1] * len(new)
    out = [0] * len(new)
    view_d = len(new) - 1
    chunk = stride[-1] if stride else 1
    tensor_numel, view_numel = 1, 1
    for d in range(len(shape) - 1, -1, -1):
        tensor_numel *= shape[d]
        if d == 0 or (shape[d - 1] != 1 and stride[d - 1] != tensor_numel * chunk):
            while view_d >= 0 and (view_numel < tensor_numel or new[view_d] == 1):
                out[view_d] = view_numel * chunk
                view_numel *= new[view_d]
                view_d -= 1
            if view_numel != tensor_numel:
                return None
            if d > 0:
                chunk = stride[d - 1]
                tensor_numel, view_numel = 1, 1
    return out if view_d == -1 else None


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
    if name == "reshape":  # a view wherever torch's view would be one (not only contiguous tensors)
        new = list(op[1])
        if -1 in new:
            known = 1
            for n in new:
                if n != -1:
                    known *= n
            new[new.index(-1)] = t.numel() // known if known else 0
        return t.view(new) if _view_stride(list(t.shape), list(t.stride()), new) is not None else None
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
    # Both sides from one traced tensor (blocks of one pool): the eager lazy copy gathers
    # first (two passes, as torch's pool[d] = pool[s]); so does the graph. Sides from
    # separate tensors that alias are refused at run time if they may overlap (UnsafeCopy).
    base = lambda x: x.t if isinstance(x, TracedLazy) else x  # noqa: E731
    one_pool = base(src) is base(dst)
    if sidx is not None:  # dst = base.index_select(dim, i).permute(p): gather into dst.permute(p^-1)
        if not identity:
            return False
        if one_pool:
            tmp = torch.empty(dt.shape, dtype=st.dtype, device=st.device)
            gather(engine, st, sidx[1], sidx[0], tmp.permute(_inverse(sperm)))
            ops.copy_(dt, tmp, engine)
        else:
            gather(engine, st, sidx[1], sidx[0], dt.permute(_inverse(sperm)))
        return True
    if didx is not None:  # base.index_select(dim, i).permute(p) = src: scatter src.permute(p^-1)
        if not identity:
            return False
        if one_pool:
            tmp = torch.empty(st.shape, dtype=st.dtype, device=st.device)
            ops.copy_(tmp, st, engine)
            st = tmp
        scatter_(engine, dt, didx[1], st.permute(_inverse(dperm)), didx[0])
        return True
    if identity:
        if decide("copy", [dt, st]) == "torch":
            dt.copy_(st)
        else:
            ops.copy_(dt, st, engine)
    else:
        lo, hi = fn.clamp or (0.0, 0.0)
        extra = (float(fn.scale), float(fn.bias), bool(fn.round), fn.clamp is not None, float(lo), float(hi))
        if decide("copy_fn", [dt, st], extra) == "torch":
            _kind("copy_fn", extra)[0](dt, st)
        else:
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
