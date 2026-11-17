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

# Static checks of copyengine's own C++ (.clang-tidy): needs clang-tidy.
tidy:
    cmake -S . -B build/tidy -G Ninja -DCE_BUILD_TESTS=ON -DCE_CUDA_ARCHS={{cuda_archs}} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cmake --build build/tidy
    python3 -c "import json; print('\n'.join(sorted({e['file'] for e in json.load(open('build/tidy/compile_commands.json')) if '/src/' in e['file'] and e['file'].endswith('.cpp')})))" | xargs -P 8 -n 1 clang-tidy -p build/tidy --quiet --warnings-as-errors='*'

# The GPU tests under compute-sanitizer: out-of-bounds and misaligned accesses
# (memcheck) and shared-memory races (racecheck) in copyengine's kernels. Needs a
# native Linux driver (under WSL2, the WDDM debugger interface enabled on Windows).
sanitize:
    SKBUILD_CMAKE_DEFINE="CE_CUDA_ARCHS={{cuda_archs}}" pip install --no-build-isolation . -q
    compute-sanitizer --tool memcheck --report-api-errors no --target-processes all --error-exitcode 1 python -m pytest -q -x tests/python/test_cuda.py tests/python/test_device_index.py tests/python/test_footprints.py tests/python/test_plan.py
    compute-sanitizer --tool racecheck --report-api-errors no --target-processes all --error-exitcode 1 python -m pytest -q -x tests/python/test_cuda.py

# Random copy programs against torch in submission order, every strategy forced
# and every mode (tests/fuzz/fuzz.py); one process per seed, failures printed.
fuzz first="0" count="100" modes="forced,explicit,scope,freeze,plan":
    SKBUILD_CMAKE_DEFINE="CE_CUDA_ARCHS={{cuda_archs}}" pip install --no-build-isolation . -q
    for s in $(seq {{first}} $(({{first}} + {{count}} - 1))); do timeout 600 python tests/fuzz/fuzz.py $s 1 {{modes}} 2>&1 | grep -E "^FAIL|Error" | head -3; done; echo "fuzzed seeds {{first}}..$(({{first}} + {{count}} - 1))"
