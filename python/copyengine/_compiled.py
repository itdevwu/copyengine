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

from . import _torch_ops

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None


def compiling() -> bool:
    return torch is not None and torch.compiler.is_compiling()


def available() -> bool:
    return _torch_ops.copy_ is not None and hasattr(torch.ops.copyengine, "quantize_")


class TracedLazy:
    """`ce.lazy(t)` while Dynamo traces: the tensor and the ops, recorded."""

    def __init__(self, t, ops=()):
        self.t, self.ops = t, ops

    def _with(self, *op):
        return TracedLazy(self.t, self.ops + (op,))

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
        ops.gather_(dt.permute(_inverse(sperm)), st, sidx[0], sidx[1], engine)
        return True
    if didx is not None:  # base.index_select(dim, i).permute(p) = src: scatter src.permute(p^-1)
        if not identity:
            return False
        ops.scatter_(dt, didx[1], st.permute(_inverse(dperm)), didx[0], engine)
        return True
    if identity:
        ops.copy_(dt, st, engine)
    else:
        lo, hi = fn.clamp or (0.0, 0.0)
        ops.copy_fn_(dt, st, engine, fn.scale, fn.bias, fn.round, fn.clamp is not None, lo, hi)
    return True


def copy_many(engine: int, dsts, srcs) -> bool:
    if not all(isinstance(t, torch.Tensor) for t in list(dsts) + list(srcs)):
        return False
    torch.ops.copyengine.copy_many_(list(dsts), list(srcs), engine)
    return True


def gather(engine: int, src, index, dim, out) -> bool:
    if not (isinstance(src, torch.Tensor) and isinstance(index, torch.Tensor) and index.is_cuda):
        return False
    torch.ops.copyengine.gather_(out, src, dim, index.to(torch.int64).contiguous(), engine)
    return True


def scatter_(engine: int, dst, index, src, dim) -> bool:
    if not (isinstance(dst, torch.Tensor) and isinstance(index, torch.Tensor) and index.is_cuda):
        return False
    torch.ops.copyengine.scatter_(dst, index.to(torch.int64).contiguous(), src, dim, engine)
    return True


def _index(ix):
    if ix is None:
        return None
    if isinstance(ix, torch.Tensor):
        return ix.to(torch.int64).contiguous()
    return None  # lists / arrays: not traced (the caller breaks the graph)


def tile(engine: int, quantize: bool, x, q, scale, zero, fmt, group, eps, gs, static, x_index, index) -> bool:
    from . import _quant
    if fmt not in _quant.FMT:
        raise ValueError(f"unknown format {fmt!r}; one of {sorted(_quant.FMT)}")
    xi, ci = _index(x_index), _index(index)
    if (x_index is not None and xi is None) or (index is not None and ci is None):
        return False
    D = x.shape[-1]
    g = _quant.UNIT.get(fmt, group if group is not None else D)
    if group is not None and fmt in _quant.UNIT and group != g:
        raise ValueError(f"{fmt} uses units of {g} values")
    if fmt == "uint4" and zero is None:
        raise ValueError("uint4 needs a zero-point tensor (`zero`)")
    z = zero if fmt == "uint4" else None
    if quantize:
        torch.ops.copyengine.quantize_(x, q, scale, z, _quant.FMT[fmt], g, float(eps), float(gs), bool(static), xi, ci,
                                       engine)
    else:
        torch.ops.copyengine.dequantize_(q, scale, x, z, _quant.FMT[fmt], g, float(gs), xi, ci, engine)
    return True
