"""einops on lazy views (docs/design/v0.5.md): `einops.rearrange` and
`einops.repeat` applied to `ce.lazy(x)` compose into the lazy view instead
of materializing, so a chain of rearrangements, gathers, conversions and the
write into a destination plans and runs as one copy:

    src = einops.rearrange(ce.lazy(kv), "(blk tok) h d -> blk h tok d", tok=16)
    ce.copy_(cache, src.index_select(0, pages))

einops finds this backend by itself (a subclass of its AbstractBackend whose
framework is loaded). What is not a view of the source stays outside:
`reduce` (a reduction), `repeat` along an axis longer than one (copies), and
concatenation raise.
"""

from __future__ import annotations

try:
    import einops
    import einops._backends as _eb
except ImportError:  # pragma: no cover - einops is optional
    einops = _eb = None


class _Recorder:
    """Shapes and view ops only: what einops does to a lazy view, recorded."""

    def __init__(self, shape, ops=()):
        self.shape, self.ops = tuple(shape), ops

    def _with(self, *op):
        from ._compiled import _shape_after
        return _Recorder(_shape_after(self.shape, op), self.ops + (op,))

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
        return self._with("reshape", tuple(shape))

    def permute(self, *dims):
        dims = dims[0] if len(dims) == 1 and isinstance(dims[0], (list, tuple)) else dims
        return self._with("permute", tuple(dims))

    def unsqueeze(self, dim):
        return self._with("unsqueeze", dim)

    def expand(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
        return self._with("expand", tuple(shape))


def _ops(fn, pattern, shape, axes):
    """The view ops einops' `fn` makes of `pattern` on a view of `shape`."""
    return getattr(einops, fn)(_Recorder(shape), pattern, **dict(axes)).ops


try:  # inside torch.compile regions: a constant of the graph (einops itself may not be traceable there)
    import torch as _torch
    _const_ops = _torch._dynamo.assume_constant_result(_ops)
except Exception:  # pragma: no cover - without torch
    _const_ops = _ops


def _apply(fn, x, pattern, axes):
    if einops is None:
        raise ImportError("copyengine.rearrange / repeat need einops")
    from . import _compiled
    if _compiled.compiling() and isinstance(x, _compiled.TracedLazy) and _compiled.static((x.shape, tuple(axes.values()))):
        for op in _const_ops(fn, pattern, tuple(x.shape), tuple(sorted(axes.items()))):
            x = x._with(*op)
        return x
    return getattr(einops, fn)(x, pattern, **axes)


def rearrange(x, pattern: str, **axes):
    """``einops.rearrange`` for lazy views, also inside torch.compile regions
    (where einops' own functions may be graph nodes no lazy view can enter)."""
    return _apply("rearrange", x, pattern, axes)


def repeat(x, pattern: str, **axes):
    """``einops.repeat`` for lazy views (see :func:`rearrange`)."""
    return _apply("repeat", x, pattern, axes)


if _eb is not None:

    class CopyEngineBackend(_eb.AbstractBackend):
        framework_name = "copyengine"

        def __init__(self):
            from ._compiled import TracedLazy
            from ._tensor import Lazy
            self._types = (Lazy, TracedLazy, _Recorder)

        def is_appropriate_type(self, tensor):
            return isinstance(tensor, self._types)

        def shape(self, x):
            return tuple(x.shape)

        def reshape(self, x, shape):
            return x.reshape(tuple(shape))

        def transpose(self, x, axes):
            return x.permute(tuple(axes))

        def add_axis(self, x, new_position):
            return x.unsqueeze(new_position)

        def tile(self, x, repeats):
            shape = list(x.shape)
            for i, r in enumerate(repeats):
                if r == 1:
                    continue
                if shape[i] != 1:
                    raise NotImplementedError(
                        "copyengine: repeating along an axis longer than one copies data, not a view")
                shape[i] = r
            return x.expand(tuple(shape)) if shape != list(x.shape) else x

        def reduce(self, x, operation, axes):
            raise NotImplementedError("copyengine: einops.reduce is a reduction; lazy views only move data")

        def stack_on_zeroth_dimension(self, tensors):
            raise NotImplementedError("copyengine: stacking lazy views is not a view")

        def concat(self, tensors, axis):
            raise NotImplementedError("copyengine: concatenating lazy views is not a view")

        def is_float_type(self, x):
            return True
