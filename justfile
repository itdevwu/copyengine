build_dir := "build/dev"

default: test

configure:
    cmake -S . -B {{build_dir}} -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo

build: configure
    cmake --build {{build_dir}}

test-cpp: build
    ctest --test-dir {{build_dir}} --output-on-failure

test-python:
    pip install --no-build-isolation . -q
    pytest -q

test: test-cpp test-python

# Dev shell with the GPU, CUDA 12.9 toolkit and this checkout mounted at /work.
docker-shell:
    docker build -t copyengine-dev docker
    docker run --rm -it --gpus all -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/work -w /work copyengine-dev
