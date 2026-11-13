cmake_minimum_required(VERSION 3.26)
project(copyengine VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CE_BUILD_TESTS "Build C++ tests" ${PROJECT_IS_TOP_LEVEL})
# scikit-build-core sets SKBUILD=2, which option() would read as OFF.
if(DEFINED SKBUILD)
  set(_ce_python_default ON)
else()
  set(_ce_python_default OFF)
endif()
option(CE_BUILD_PYTHON "Build the Python extension" ${_ce_python_default})

configure_file(src/core/config.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/generated/copyengine/config.hpp @ONLY)

# Planner core + C ABI. Platform backends are added as separate sources later.
add_library(copyengine
  src/core/version.cpp
  src/capi/capi.cpp
)
add_library(copyengine::copyengine ALIAS copyengine)
target_include_directories(copyengine
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  PRIVATE src
)
target_compile_options(copyengine PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror=return-type>
)
set_target_properties(copyengine PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(copyengine PRIVATE CE_BUILDING_LIBRARY)

if(CE_BUILD_TESTS)
  enable_testing()
  add_executable(ce_test_smoke tests/cpp/test_smoke.cpp)
  target_link_libraries(ce_test_smoke PRIVATE copyengine::copyengine)
  add_test(NAME smoke COMMAND ce_test_smoke)
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
