cmake_minimum_required(VERSION 3.20)
project(pptree VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Platform detection for golden files
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(PPTREE_OS "linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(PPTREE_OS "macos")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(PPTREE_OS "windows")
else()
  set(PPTREE_OS "unknown")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(PPTREE_COMPILER "gcc")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(PPTREE_COMPILER "clang")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(PPTREE_COMPILER "msvc")
else()
  set(PPTREE_COMPILER "unknown")
endif()

set(PPTREE_PLATFORM "${PPTREE_OS}-${PPTREE_COMPILER}")

option(PPTREE_DOUBLE_PRECISION "Use double precision for feature data" OFF)
option(PPTREE_CORE_ONLY "Build only the core library, skip CLI, tests, and benchmarks" OFF)
option(PPTREE_NATIVE_ARCH "Use -march=native for CPU-specific optimization" ON)
if(PPTREE_DOUBLE_PRECISION)
  add_compile_definitions(PPTREE_DOUBLE_PRECISION)
endif()

add_compile_definitions(EIGEN_DONT_PARALLELIZE)
add_compile_definitions(EIGEN_NO_AUTOMATIC_RESIZING)

if(WIN32)
  # Disable vectorization to prevent alignment issues on Windows
  add_compile_definitions(EIGEN_DONT_VECTORIZE)
endif()

if(PPTREE_NATIVE_ARCH AND NOT CMAKE_BUILD_TYPE MATCHES Debug)
  add_compile_options(-march=native)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  add_compile_options(-O0)
  add_compile_options(-g)
  add_compile_definitions(_GLIBCXX_DEBUG)
  add_compile_definitions(_GLIBCXX_ASSERTIONS)
  add_compile_definitions(EIGEN_INITIALIZE_MATRICES_BY_NAN)
  add_compile_definitions(EIGEN_INTERNAL_DEBUGGING)
  add_compile_definitions(EIGEN_RUNTIME_NO_MALLOC)
  add_compile_options(-fno-omit-frame-pointer)

  add_link_options(--coverage)

  if(UNIX AND NOT APPLE)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
  endif()

  add_link_options(--coverage)

  if(WIN32)
    add_compile_options(-Wa,-mbig-obj)
  endif()
endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
  add_compile_options(-O3)
  add_compile_definitions(NDEBUG)
  add_compile_definitions(EIGEN_NO_DEBUG)
endif()

if(WIN32)
  add_link_options(-static)
endif()

# Add search hints
list(APPEND CMAKE_PREFIX_PATH
  /usr/local/opt/libomp
  /opt/homebrew/opt/libomp)

# Find OpenMP
find_package(OpenMP)

# Find Jemalloc using pkg-config
include(FindPkgConfig)
pkg_check_modules(Jemalloc IMPORTED_TARGET jemalloc)

if(NOT OpenMP_FOUND)
  message(WARNING "Could NOT find OpenMP. Parallelization will not be available.")
endif()

include(Dependencies.cmake)

# ---------------------------------------------------------------------------
# Common compile interface for all pptree OBJECT libraries
# ---------------------------------------------------------------------------
add_library(pptree-compile INTERFACE)

target_include_directories(pptree-compile INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(pptree-compile INTERFACE
  Eigen3::Eigen
  pcg_lib)

if(OpenMP_FOUND)
  target_link_libraries(pptree-compile INTERFACE OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# OBJECT libraries (one per subfolder)
# ---------------------------------------------------------------------------
add_subdirectory(src/models)
add_subdirectory(src/stats)
add_subdirectory(src/serialization)
add_subdirectory(src/utils)
add_subdirectory(src/io)

# ---------------------------------------------------------------------------
# Core library (composed from OBJECT libraries)
# ---------------------------------------------------------------------------
add_library(pptree-core
  $<TARGET_OBJECTS:pptree-models>
  $<TARGET_OBJECTS:pptree-stats>
  $<TARGET_OBJECTS:pptree-utils>
  $<TARGET_OBJECTS:pptree-io>)

target_link_libraries(pptree-core PUBLIC pptree-compile)

if(Jemalloc_FOUND)
  message(STATUS "Jemalloc found: ${Jemalloc_INCLUDE_DIRS}")
  target_link_libraries(pptree-core PRIVATE PkgConfig::Jemalloc)
else()
  message(WARNING "Jemalloc not found. Program could be unstable on highly multi-threaded operations.")
endif()

if(NOT PPTREE_CORE_ONLY)
  # ---------------------------------------------------------------------------
  # CLI
  # ---------------------------------------------------------------------------
  add_subdirectory(src/cli)

  add_executable(pptree src/cli/CLI.cpp $<TARGET_OBJECTS:pptree-cli>)
  target_include_directories(pptree PRIVATE ${csv_SOURCE_DIR}/include)
  target_link_libraries(pptree pptree-core pptree-serialization nlohmann_json::nlohmann_json csv CLI11::CLI11 fmt::fmt)
  target_compile_definitions(pptree PRIVATE PPTREE_VERSION="${PROJECT_VERSION}")

  # ---------------------------------------------------------------------------
  # Tests
  # ---------------------------------------------------------------------------
  enable_testing()

  set(PPTREE_TEST_SRC
    src/test.cpp
    src/models/Tree.test.cpp
    src/models/Forest.test.cpp
    src/models/PPStrategy.test.cpp
    src/models/VariableImportance.test.cpp
    src/stats/Stats.test.cpp
    src/stats/GroupPartition.test.cpp
    src/stats/ConfusionMatrix.test.cpp
    src/stats/Simulation.test.cpp
    src/serialization/Json.test.cpp
    src/utils/Math.test.cpp
    src/utils/Invariant.test.cpp
    src/cli/CLIOptions.test.cpp
    src/cli/CLI.integration.test.cpp
    src/golden/Reproducibility.test.cpp
    src/golden/Threading.test.cpp
    src/cli/VarsSpec.test.cpp
    src/cli/Benchmark.test.cpp
    src/cli/BenchmarkReport.test.cpp)

  add_executable(pptree-test ${PPTREE_TEST_SRC} $<TARGET_OBJECTS:pptree-cli>)

  target_link_libraries(pptree-test
    pptree-core
    pptree-serialization
    nlohmann_json::nlohmann_json
    gtest
    CLI11::CLI11
    csv
    fmt::fmt)

  target_include_directories(pptree-test PRIVATE
    ${csv_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR})

  target_compile_definitions(pptree-test PRIVATE
    PPTREE_BINARY_PATH="${CMAKE_BINARY_DIR}/pptree"
    PPTREE_DATA_DIR="${CMAKE_SOURCE_DIR}/../data"
    PPTREE_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/../golden"
    PPTREE_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/../fixtures"
    PPTREE_BENCH_SCENARIOS="${CMAKE_SOURCE_DIR}/../bench/default-scenarios.json"
    PPTREE_PLATFORM="${PPTREE_PLATFORM}"
    PPTREE_VERSION="${PROJECT_VERSION}")

  add_dependencies(pptree-test pptree)

  include(GoogleTest)
  gtest_discover_tests(pptree-test DISCOVERY_MODE PRE_TEST)

  # ---------------------------------------------------------------------------
  # Golden file generator
  # ---------------------------------------------------------------------------
  add_executable(pptree-golden-gen src/golden/GoldenGen.cpp)

  target_link_libraries(pptree-golden-gen
    pptree-core
    pptree-serialization
    nlohmann_json::nlohmann_json
    csv
    fmt::fmt)

  target_include_directories(pptree-golden-gen PRIVATE
    ${csv_SOURCE_DIR}/include)

  target_compile_definitions(pptree-golden-gen PRIVATE
    PPTREE_DATA_DIR="${CMAKE_SOURCE_DIR}/../data"
    PPTREE_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/../golden"
    PPTREE_PLATFORM="${PPTREE_PLATFORM}")
endif()
