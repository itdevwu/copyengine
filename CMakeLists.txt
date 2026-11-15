cmake_minimum_required(VERSION 3.26)
project(copyengine VERSION 0.4.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CE_CUDA "AUTO" CACHE STRING "Build the CUDA backend: ON, OFF or AUTO (if the toolkit headers are found)")
option(CE_BUILD_TESTS "Build C++ tests" ${PROJECT_IS_TOP_LEVEL})
# scikit-build-core sets SKBUILD=2, which option() would read as OFF.
if(DEFINED SKBUILD)
  set(_ce_python_default ON)
else()
  set(_ce_python_default OFF)
endif()
option(CE_BUILD_PYTHON "Build the Python extension" ${_ce_python_default})

# Pre-release builds (".dev1", "rc1") append a suffix from the tag; the
# version itself lives only in project() above.
set(CE_VERSION_SUFFIX "" CACHE STRING "pre-release suffix of the version string, e.g. .dev1")
configure_file(src/core/config.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/generated/copyengine/config.hpp @ONLY)

# Planner core, host backend and C ABI. Platform backends are added below.
add_library(copyengine
  src/core/version.cpp
  src/core/probe.cpp
  src/core/ir.cpp
  src/core/types.cpp
  src/core/plan.cpp
  src/core/tuner.cpp
  src/core/engine.cpp
  src/core/frontend.cpp
  src/backends/host/host.cpp
  src/capi/capi.cpp
)
add_library(copyengine::copyengine ALIAS copyengine)
# Host and device conversions must agree bit for bit: no FMA contraction.
set_source_files_properties(src/backends/host/host.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
# The quantization test hooks evaluate convert.hpp / quant.hpp too.
set_source_files_properties(src/python/module.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
find_package(Threads REQUIRED)
target_link_libraries(copyengine PRIVATE Threads::Threads)
target_include_directories(copyengine
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  PRIVATE src
)
option(CE_WERROR "treat compiler warnings as errors (CI)" OFF)
target_compile_options(copyengine PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror=return-type>
  $<$<AND:$<BOOL:${CE_WERROR}>,$<CXX_COMPILER_ID:GNU,Clang>>:-Werror>
  # GCC 13 false positive on std::variant holding std::shared_ptr (core/json.hpp)
  $<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized>
)
set_target_properties(copyengine PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(copyengine PRIVATE CE_BUILDING_LIBRARY)

# CUDA backend: needs only the toolkit headers. The driver is dlopen'ed at
# runtime, so nothing links against libcuda or cudart.
set(CE_WITH_CUDA OFF)
if(NOT CE_CUDA STREQUAL "OFF")
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(CE_WITH_CUDA ON)
  elseif(CE_CUDA STREQUAL "ON")
    message(FATAL_ERROR "CE_CUDA=ON but the CUDA toolkit was not found")
  endif()
endif()
set(CE_CUDA_KERNELS OFF)
if(CE_WITH_CUDA)
  target_sources(copyengine PRIVATE src/backends/cuda/driver.cpp src/backends/cuda/probe.cpp
    src/backends/cuda/executor.cpp)
  target_include_directories(copyengine SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(copyengine PRIVATE ${CMAKE_DL_LIBS})

  # Copy kernels: compiled AOT to a fatbin with nvcc and embedded as a byte
  # array; loaded at runtime with cuModuleLoadData (no cudart dependency).
  set(CE_CUDA_ARCHS "75;80;86;89;90;100;120" CACHE STRING
      "SM architectures for the kernel fatbin (PTX of the last one is added), or 'native'")
  find_program(CE_NVCC nvcc HINTS ${CUDAToolkit_BIN_DIR})
  if(CE_NVCC)
    set(CE_CUDA_KERNELS ON)
    if(CE_CUDA_ARCHS STREQUAL "native")
      set(_gencode -arch=native)
    else()
      set(_gencode)
      foreach(a IN LISTS CE_CUDA_ARCHS)
        list(APPEND _gencode -gencode arch=compute_${a},code=sm_${a})
        set(_last ${a})
      endforeach()
      list(APPEND _gencode -gencode arch=compute_${_last},code=compute_${_last})
    endif()
    set(_fatbin ${CMAKE_CURRENT_BINARY_DIR}/ce_kernels.fatbin)
    set(_embed ${CMAKE_CURRENT_BINARY_DIR}/ce_kernels_fatbin.c)
    add_custom_command(
      OUTPUT ${_fatbin}
      # compress-mode=size: ~27x smaller; needs a CUDA 12.4+ era driver to load,
      # older drivers fail kernel admission with a clear reason.
      COMMAND ${CE_NVCC} --fatbin -std=c++17 -O3 --fmad=false --compress-mode=size ${_gencode}
              -I${CMAKE_CURRENT_SOURCE_DIR}/src -o ${_fatbin} ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cuda/kernels.cu
      DEPENDS src/backends/cuda/kernels.cu src/core/convert.hpp
      COMMENT "Compiling copy kernels (${CE_CUDA_ARCHS})"
      VERBATIM)
    add_custom_command(
      OUTPUT ${_embed}
      COMMAND ${CMAKE_COMMAND} -DINPUT=${_fatbin} -DOUTPUT=${_embed} -DNAME=ce_kernels_fatbin
              -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
      DEPENDS ${_fatbin} cmake/embed.cmake
      VERBATIM)
    target_sources(copyengine PRIVATE ${_embed})
  else()
    message(WARNING "nvcc not found: CUDA conversions and kernel strategies will be unavailable")
  endif()
endif()
target_compile_definitions(copyengine PRIVATE CE_CUDA_KERNELS=$<BOOL:${CE_CUDA_KERNELS}>)
target_compile_definitions(copyengine PRIVATE CE_WITH_CUDA=$<BOOL:${CE_WITH_CUDA}>)
message(STATUS "copyengine: CUDA backend ${CE_WITH_CUDA}, kernels ${CE_CUDA_KERNELS}")

if(CE_BUILD_TESTS)
  enable_testing()
  add_executable(ce_test_smoke tests/cpp/test_smoke.cpp)
  target_link_libraries(ce_test_smoke PRIVATE copyengine::copyengine)
  add_test(NAME smoke COMMAND ce_test_smoke)
  add_executable(ce_test_probe tests/cpp/test_probe.cpp)
  target_link_libraries(ce_test_probe PRIVATE copyengine::copyengine)
  add_test(NAME probe COMMAND ce_test_probe)
  add_executable(ce_test_capi tests/cpp/test_capi.c)
  target_link_libraries(ce_test_capi PRIVATE copyengine::copyengine)
  add_test(NAME capi COMMAND ce_test_capi)
  add_executable(ce_test_convert tests/cpp/test_convert.cpp)
  target_link_libraries(ce_test_convert PRIVATE copyengine::copyengine)
  target_include_directories(ce_test_convert PRIVATE src)
  target_compile_options(ce_test_convert PRIVATE -ffp-contract=off)
  add_test(NAME convert COMMAND ce_test_convert)
endif()

if(CE_BUILD_PYTHON)
  find_package(Python 3.10 REQUIRED
    COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_core STABLE_ABI NB_STATIC src/python/module.cpp)
  target_link_libraries(_core PRIVATE copyengine)
  target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)  # quant.hpp test hooks
  install(TARGETS _core LIBRARY DESTINATION copyengine)

  # Interception shim sources, laid out as in the wheel: the runtime compiles
  # them for any torch the prebuilt shim below does not match.
  set(_shim_src ${CMAKE_CURRENT_BINARY_DIR}/_shim_src)
  configure_file(src/torch/shim.cpp ${_shim_src}/shim.cpp COPYONLY)
  foreach(h frontend engine ir types)
    configure_file(include/copyengine/${h}.hpp ${_shim_src}/include/copyengine/${h}.hpp COPYONLY)
  endforeach()
  install(DIRECTORY ${_shim_src} DESTINATION copyengine)

  # copyengine._torch_stable: explicit APIs as torch custom ops on the stable
  # libtorch ABI (torch >= 2.10), one binary for every later torch. Optional:
  # without it the Python layer issues copies itself.
  set(CE_TORCH "AUTO" CACHE STRING "Build the stable-ABI torch extension: ON, OFF or AUTO (if torch >= 2.10 imports)")
  if(NOT CE_TORCH STREQUAL "OFF")
    string(CONCAT _q "import torch; from torch.utils import cpp_extension as c; "
           "v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); "
           "assert v >= (2, 10), torch.__version__; ")
    execute_process(COMMAND "${Python_EXECUTABLE}" -c "${_q}print(';'.join(c.include_paths()), end='')"
                    RESULT_VARIABLE _torch_rc OUTPUT_VARIABLE _torch_inc ERROR_VARIABLE _torch_err)
    execute_process(COMMAND "${Python_EXECUTABLE}" -c "${_q}print(';'.join(c.library_paths()), end='')"
                    OUTPUT_VARIABLE _torch_lib ERROR_QUIET)
    if(_torch_rc EQUAL 0)
      find_library(CE_TORCH_CPU torch_cpu PATHS ${_torch_lib} NO_DEFAULT_PATH)
      find_library(CE_TORCH_C10 c10 PATHS ${_torch_lib} NO_DEFAULT_PATH)
      add_library(copyengine_torch MODULE src/torch/ops.cpp)
      target_include_directories(copyengine_torch SYSTEM PRIVATE ${_torch_inc})
      target_include_directories(copyengine_torch PRIVATE $<TARGET_PROPERTY:copyengine,INTERFACE_INCLUDE_DIRECTORIES>)
      # Only the stable ABI as of torch 2.10 may be used.
      target_compile_definitions(copyengine_torch PRIVATE TORCH_TARGET_VERSION=0x020A000000000000)
      target_link_libraries(copyengine_torch PRIVATE ${CE_TORCH_CPU} ${CE_TORCH_C10} ${CMAKE_DL_LIBS})
      set_target_properties(copyengine_torch PROPERTIES PREFIX "" OUTPUT_NAME "_torch_stable"
                            CXX_VISIBILITY_PRESET hidden)
      install(TARGETS copyengine_torch LIBRARY DESTINATION copyengine)
      message(STATUS "copyengine: stable-ABI torch extension ON")

      # The interception shim, prebuilt for the build-time torch, under the key
      # the runtime looks up (python/copyengine/_shim_key.py). CI adds shims
      # for other torch releases next to it.
      execute_process(COMMAND "${Python_EXECUTABLE}" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI), end='')"
                      OUTPUT_VARIABLE _torch_abi)
      execute_process(COMMAND "${Python_EXECUTABLE}" ${CMAKE_CURRENT_SOURCE_DIR}/python/copyengine/_shim_key.py
                              ${_shim_src} ${PROJECT_VERSION}${CE_VERSION_SUFFIX}
                      OUTPUT_VARIABLE _shim_key RESULT_VARIABLE _shim_key_rc)
      find_library(CE_TORCH_LIB torch PATHS ${_torch_lib} NO_DEFAULT_PATH)
      if(_shim_key_rc EQUAL 0)
        add_library(copyengine_shim MODULE ${_shim_src}/shim.cpp)
        target_include_directories(copyengine_shim SYSTEM PRIVATE ${_torch_inc})
        target_include_directories(copyengine_shim PRIVATE ${_shim_src}/include)
        target_compile_definitions(copyengine_shim PRIVATE _GLIBCXX_USE_CXX11_ABI=${_torch_abi})
        target_link_libraries(copyengine_shim PRIVATE ${CE_TORCH_C10} ${CE_TORCH_CPU} ${CE_TORCH_LIB} ${CMAKE_DL_LIBS})
        set_target_properties(copyengine_shim PROPERTIES PREFIX "" OUTPUT_NAME "libcopyengine_shim" CXX_VISIBILITY_PRESET hidden)
        install(TARGETS copyengine_shim LIBRARY DESTINATION copyengine/shims/${_shim_key})
      endif()
    elseif(CE_TORCH STREQUAL "ON")
      message(FATAL_ERROR "CE_TORCH=ON but torch >= 2.10 is not importable: ${_torch_err}")
    else()
      message(STATUS "copyengine: stable-ABI torch extension OFF (torch >= 2.10 not importable at build time)")
    endif()
  endif()
endif()
