cmake_minimum_required(VERSION 3.20)
project(pptree VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_definitions(EIGEN_DONT_PARALLELIZE)
add_compile_definitions(EIGEN_NO_AUTOMATIC_RESIZING)
add_compile_options(-march=native)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  add_compile_options(-O0)
  add_compile_options(-g)
  add_compile_definitions(_GLIBCXX_DEBUG)
  add_compile_definitions(_GLIBCXX_ASSERTIONS)
  add_compile_definitions(EIGEN_INITIALIZE_MATRICES_BY_NAN)
  add_compile_definitions(EIGEN_INTERNAL_DEBUGGING)
  add_compile_definitions(EIGEN_RUNTIME_NO_MALLOC)
  add_compile_options(-fno-omit-frame-pointer)


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
  GIT_TAG 3.4.0
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

# Collect all .cpp files from the specified directories
file(GLOB_RECURSE ALL_CPP_FILES ./**/*.cpp)

# Collect all .cpp files that are not test files
set(NON_TEST_CPP_FILES "")
foreach(file ${ALL_CPP_FILES})
  if(NOT file MATCHES "\\.test\\.cpp$" AND NOT file MATCHES "\\CLI\\.cpp$" AND NOT file MATCHES "Sandbox\\.cpp$")
    list(APPEND NON_TEST_CPP_FILES ${file})
  endif()
endforeach()

# Add library with all non-test files
add_library(pptree-core ${NON_TEST_CPP_FILES})

target_include_directories(pptree-core
  PUBLIC include
  PUBLIC src)

target_link_libraries(pptree-core
  Eigen3::Eigen
  nlohmann_json::nlohmann_json
  pcg_lib)

if(OpenMP_FOUND)
  target_link_libraries(pptree-core OpenMP::OpenMP_CXX)
endif()

if(Jemalloc_FOUND)
  message(STATUS "Jemalloc found: ${Jemalloc_INCLUDE_DIRS}")
  target_link_libraries(pptree-core PkgConfig::Jemalloc)
else()
  message(WARNING "Jemalloc not found. Program could be unstable on high multi-threaded operations.")
endif()

# Test runner
enable_testing()

# Collect all .test.cpp
file(GLOB_RECURSE TEST_SOURCES ./**/*.test.cpp)

if(TEST_SOURCES AND NOT PPTREE_SKIP_TESTS)
  add_library(pptree_singlethreaded ${NON_TEST_CPP_FILES})

  target_link_libraries(pptree_singlethreaded
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    pcg_lib)

  # Google Test
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
  )
  FetchContent_MakeAvailable(googletest)

  # Add executable for the tests
  add_executable(pptree-test ${TEST_SOURCES})

  target_link_libraries(pptree-test
    pptree_singlethreaded
    gtest_main
    gtest
    CLI11::CLI11
    csv)

  target_include_directories(pptree-test
    PUBLIC src
    PUBLIC include
    PUBLIC ${gtest_SOURCE_DIR}/include
    PUBLIC ${gtest_SOURCE_DIR}
    PUBLIC ${csv_SOURCE_DIR}/include)

  target_compile_definitions(pptree-test PRIVATE
    PPTREE_BINARY_PATH="${CMAKE_BINARY_DIR}/pptree"
    PPTREE_DATA_DIR="${CMAKE_SOURCE_DIR}/../data")

  add_dependencies(pptree-test pptree)

  include(GoogleTest)
  gtest_discover_tests(pptree-test)
endif()


# Fetch csv-parser
FetchContent_Declare(
  csv
  GIT_REPOSITORY https://github.com/vincentlaucsb/csv-parser.git
  GIT_SHALLOW TRUE
  GIT_TAG 2.2.1
)

FetchContent_MakeAvailable(csv)

# CLI11 for command line parsing
FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.6.1
)
FetchContent_MakeAvailable(cli11)

# Setup cli executable
add_executable(pptree src/CLI.cpp)
target_include_directories(pptree PRIVATE ${csv_SOURCE_DIR}/include)
target_link_libraries(pptree pptree-core csv CLI11::CLI11)


