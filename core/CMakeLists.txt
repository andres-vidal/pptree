cmake_minimum_required(VERSION 3.14)
project(pptree VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -D_GLIBCXX_ASSERTIONS -D_GLIBCXX_DEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DEIGEN_NO_DEBUG -DEIGEN_DONT_PARALLELIZE ")

# Configure Address Sanitizer in Unix systems
if(CMAKE_BUILD_TYPE MATCHES Debug AND NOT WIN32)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined")
  add_compile_options(-fsanitize=address,undefined)
  add_link_options(-fsanitize=address,undefined)
endif()

# Append Release flags to CMAKE_CXX_FLAGS
if(CMAKE_BUILD_TYPE MATCHES Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# Append Debug flags to CMAKE_CXX_FLAGS
if(CMAKE_BUILD_TYPE MATCHES Debug)
  if(WIN32)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wa,-mbig-obj")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
  endif()
endif()

# Add search hints
list(APPEND CMAKE_PREFIX_PATH 
  /usr/local/opt/libomp
  /opt/homebrew/opt/libomp)

# Find OpenMP
find_package(OpenMP)

if (NOT OpenMP_FOUND)
  message(WARNING "Could NOT find OpenMP. Parallelization will not be available.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
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

# Collect all .cpp files from the specified directories
file(GLOB_RECURSE ALL_CPP_FILES ./**/*.cpp)

# Collect all .cpp files that are not test files
set(NON_TEST_CPP_FILES "")
foreach(file ${ALL_CPP_FILES})
  if (NOT file MATCHES "\\.test\\.cpp$" AND NOT file MATCHES "\\Profiler\\.cpp$")
    list(APPEND NON_TEST_CPP_FILES ${file})
  endif()
endforeach()

# Add library with all non-test files
add_library(pptree ${NON_TEST_CPP_FILES})

target_include_directories(pptree
  PUBLIC include
  PUBLIC src)

target_link_libraries(pptree
  Eigen3::Eigen
  nlohmann_json::nlohmann_json)

if(OpenMP_FOUND)
  target_link_libraries(pptree OpenMP::OpenMP_CXX)
endif()

# Test runner
enable_testing()

# Collect all .test.cpp
file(GLOB_RECURSE TEST_SOURCES ./**/*.test.cpp)

if(TEST_SOURCES AND NOT PPTREE_SKIP_TESTS)
  add_library(pptree_singlethreaded ${NON_TEST_CPP_FILES})
  
  target_link_libraries(pptree_singlethreaded
    Eigen3::Eigen
    nlohmann_json::nlohmann_json)

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
    gtest)

  target_include_directories(pptree-test
    PUBLIC src
    PUBLIC ${gtest_SOURCE_DIR}/include
    PUBLIC ${gtest_SOURCE_DIR})

  include(GoogleTest)
  gtest_discover_tests(pptree-test)
endif()

# Add profiler executable and its dependencies
# Disable coverage for csv-parser on all platforms
set(CSV_PARSER_COVERAGE OFF CACHE BOOL "Disable coverage for csv-parser" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Disable csv-parser tests" FORCE)

FetchContent_Declare(
  csv  
  GIT_REPOSITORY https://github.com/vincentlaucsb/csv-parser.git
  GIT_SHALLOW TRUE 
  GIT_TAG 2.2.1
) 

FetchContent_GetProperties(csv)
if(NOT csv_POPULATED)
    FetchContent_Populate(csv)
    add_library(csv INTERFACE)
    target_include_directories(csv INTERFACE ${csv_SOURCE_DIR}/single_include)
endif()

add_executable(pptree-profile src/Profiler.cpp)
target_include_directories(pptree-profile PRIVATE ${csv_SOURCE_DIR}/single_include)
target_link_libraries(pptree-profile pptree csv)