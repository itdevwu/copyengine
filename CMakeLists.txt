cmake_minimum_required(VERSION 3.26)
project(copyengine VERSION 0.1.0 LANGUAGES C CXX)

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
  src/backends/host/host.cpp
  src/capi/capi.cpp
)
add_library(copyengine::copyengine ALIAS copyengine)
# Host and device conversions must agree bit for bit: no FMA contraction.
set_source_files_properties(src/backends/host/host.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
find_package(Threads REQUIRED)
target_link_libraries(copyengine PRIVATE Threads::Threads)
target_include_directories(copyengine
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  PRIVATE src
)
target_compile_options(copyengine PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror=return-type>
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
      COMMAND ${CE_NVCC} --fatbin -std=c++17 -O3 --fmad=false ${_gencode}
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
  install(TARGETS _core LIBRARY DESTINATION copyengine)
endif()
