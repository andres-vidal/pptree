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

add_library(pptree
  src/DMatrix.cpp
  src/DRStrategy.cpp
  src/Forest.cpp
  src/Group.cpp
  src/PPStrategy.cpp
  src/Tree.cpp)

target_include_directories(pptree
  PUBLIC include
  PUBLIC src)

target_link_libraries(pptree
  Eigen3::Eigen
  nlohmann_json::nlohmann_json)

# Test runner
enable_testing()

add_executable(pptree-test
  src/BootstrapDataSpec.test.cpp 
  src/Data.test.cpp
  src/DataColumn.test.cpp
  src/DataSpec.test.cpp
  src/DMatrix.test.cpp
  src/DVector.test.cpp
  src/Group.test.cpp
  src/Projector.test.cpp
  src/PPStrategy.test.cpp
  src/Tree.test.cpp
  src/Forest.test.cpp)

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