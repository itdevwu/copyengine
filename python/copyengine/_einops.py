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
    import einops._backends as _eb
except ImportError:  # pragma: no cover - einops is optional
    _eb = None


if _eb is not None:

    class CopyEngineBackend(_eb.AbstractBackend):
        framework_name = "copyengine"

        def __init__(self):
            from ._compiled import TracedLazy
            from ._tensor import Lazy
            self._types = (Lazy, TracedLazy)

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
