cmake_minimum_required(VERSION 3.20)
project(pptree VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PPTREE_DOUBLE_PRECISION "Use double precision for feature data" OFF)
if(PPTREE_DOUBLE_PRECISION)
  add_compile_definitions(PPTREE_DOUBLE_PRECISION)
endif()

add_compile_definitions(EIGEN_DONT_PARALLELIZE)
add_compile_definitions(EIGEN_NO_AUTOMATIC_RESIZING)

if(WIN32)
  # Disable vectorization to prevent alignment issues on Windows
  add_compile_definitions(EIGEN_DONT_VECTORIZE)

  if(NOT CMAKE_BUILD_TYPE MATCHES Debug)
    # Disable native `march=native` on Windows/Debug due to issues with the fmt library
    add_compile_options(-march=native)
  endif()
else()
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

include(FetchContent)

# Eigen
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.1
)
FetchContent_MakeAvailable(eigen)

# nlohmann_json
FetchContent_Declare(
  json
  URL
  https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

# PCG Random Number Generation
FetchContent_Declare(
  pcg
  GIT_REPOSITORY https://github.com/imneme/pcg-cpp.git
  GIT_TAG v0.98.1
)
FetchContent_MakeAvailable(pcg)

# Create interface library for PCG
add_library(pcg_lib INTERFACE)
target_include_directories(pcg_lib INTERFACE ${pcg_SOURCE_DIR}/include)

# Fetch csv-parser
FetchContent_Declare(
  csv
  GIT_REPOSITORY https://github.com/vincentlaucsb/csv-parser.git
  GIT_SHALLOW TRUE
  GIT_TAG 2.5.1
)
FetchContent_MakeAvailable(csv)

# CLI11 for command line parsing
FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.6.1
)
FetchContent_MakeAvailable(cli11)

# fmt for formatted output and color
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 12.1.0
)
FetchContent_MakeAvailable(fmt)

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

# ---------------------------------------------------------------------------
# Core library (composed from OBJECT libraries)
# ---------------------------------------------------------------------------
add_library(pptree-core
  $<TARGET_OBJECTS:pptree-models>
  $<TARGET_OBJECTS:pptree-stats>
  $<TARGET_OBJECTS:pptree-utils>)

target_link_libraries(pptree-core PUBLIC pptree-compile)

if(Jemalloc_FOUND)
  message(STATUS "Jemalloc found: ${Jemalloc_INCLUDE_DIRS}")
  target_link_libraries(pptree-core PRIVATE PkgConfig::Jemalloc)
else()
  message(WARNING "Jemalloc not found. Program could be unstable on highly multi-threaded operations.")
endif()

# ---------------------------------------------------------------------------
# CLI executable
# ---------------------------------------------------------------------------
add_executable(pptree src/cli/CLI.cpp)
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
  src/utils/Math.test.cpp
  src/utils/Invariant.test.cpp
  src/cli/CLIOptions.test.cpp
  src/cli/CLI.integration.test.cpp)

# Google Test
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

add_executable(pptree-test ${PPTREE_TEST_SRC})

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
  PPTREE_VERSION="${PROJECT_VERSION}")

add_dependencies(pptree-test pptree)

include(GoogleTest)
gtest_discover_tests(pptree-test DISCOVERY_MODE PRE_TEST)
