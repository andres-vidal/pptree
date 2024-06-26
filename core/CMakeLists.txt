cmake_minimum_required(VERSION 3.14)
project(pptree VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DEIGEN_NO_DEBUG")

if(CMAKE_BUILD_TYPE MATCHES Debug AND NOT WIN32)
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
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
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# Google Test
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# Collect all .cpp files from the specified directories
file(GLOB_RECURSE ALL_CPP_FILES ./**/*.cpp)

# Collect all .cpp files that are not test files
set(NON_TEST_CPP_FILES "")
foreach(file ${ALL_CPP_FILES})
  if (NOT file MATCHES "\\.test\\.cpp$")
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

# Test runner
enable_testing()

# Collect all .test.cpp
file(GLOB_RECURSE TEST_SOURCES ./**/*.test.cpp)

# Add executable for the tests
add_executable(pptree-test ${TEST_SOURCES})

target_link_libraries(pptree-test
  pptree
  gtest_main
  gtest)

target_include_directories(pptree-test
  PUBLIC src
  PUBLIC ${gtest_SOURCE_DIR}/include
  PUBLIC ${gtest_SOURCE_DIR})

include(GoogleTest)
gtest_discover_tests(pptree-test)