# copyengine

A software copy planner for ML training and inference infra. It batches, fuses
and schedules tensor copies (gather / scatter, H2D / D2H, layout changes, casts).
Despite the name, it is a library, not the GPU's DMA copy-engine hardware.

```bash
pip install .            # Python package (C++ core via nanobind)
just test                # C++ and Python tests
```

Licensed under Apache-2.0.
