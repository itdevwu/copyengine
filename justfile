build_dir := "build/dev"
# Fast local builds compile kernels for this machine's GPU only; wheels use the full list.
cuda_archs := env_var_or_default("CE_CUDA_ARCHS", "native")
werror := env_var_or_default("CE_WERROR", "OFF")

default: test

configure:
    cmake -S . -B {{build_dir}} -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCE_CUDA_ARCHS={{cuda_archs}} -DCE_WERROR={{werror}}

build: configure
    cmake --build {{build_dir}}

test-cpp: build
    ctest --test-dir {{build_dir}} --output-on-failure

test-python:
    SKBUILD_CMAKE_DEFINE="CE_CUDA_ARCHS={{cuda_archs}}" pip install --no-build-isolation . -q
    pytest -q

test: test-cpp test-python

# The GPU suite again under the configurations users run: Python interception,
# copyengine's memory from the driver instead of torch, torch's expandable
# segments. Run after `just test` on each GPU generation before a release.
test-gpu-matrix:
    CE_SHIM=0 pytest -q
    CE_TORCH_ALLOCATOR=0 pytest -q
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pytest -q

# Dev shell with the GPU, CUDA 12.9 toolkit and this checkout mounted at /work.
docker-shell:
    docker build -t copyengine-dev docker
    docker run --rm -it --gpus all -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/work -w /work copyengine-dev
