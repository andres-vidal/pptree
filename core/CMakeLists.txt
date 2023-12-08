cmake_minimum_required(VERSION 3.5)
 
project(pptree VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CSS_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

add_library(pptree
  src/pptree.cpp
  src/stats.cpp
  src/linear_algebra.cpp
  src/pp.cpp)

find_package(Eigen3 REQUIRED)

target_include_directories(pptree
  PUBLIC include)

target_link_libraries(pptree
  Eigen3::Eigen)


# Test runner
enable_testing()

add_executable(pptree-test
  test/stats.test.cpp
  test/linear_algebra.test.cpp)

find_package(GTest REQUIRED)

target_link_libraries(pptree-test
  pptree
  gtest::gtest)

target_include_directories(pptree-test
  PUBLIC src)

include(GoogleTest)
gtest_discover_tests(pptree-test)

