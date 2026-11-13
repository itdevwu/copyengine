# Development image. Pick a toolkit no newer than the host driver supports:
#   docker build --build-arg CUDA_VERSION=12.9.1 -t copyengine-dev docker
ARG CUDA_VERSION=12.9.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build g++ git python3 python3-dev python3-venv just \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -U pip \
    && /opt/venv/bin/pip install --no-cache-dir scikit-build-core nanobind pytest numpy

# Optional: PyTorch for the torch-facing tests and benchmarks.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG WITH_TORCH=1
RUN if [ "$WITH_TORCH" = "1" ]; then \
      /opt/venv/bin/pip install --no-cache-dir torch --index-url "$TORCH_INDEX"; \
    fi \
    && chmod -R a+rwX /opt/venv
ENV PATH=/opt/venv/bin:$PATH
