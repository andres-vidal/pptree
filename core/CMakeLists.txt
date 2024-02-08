cmake_minimum_required(VERSION 3.5)
 
project(pptree VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)

add_library(pptree
  src/pptree.cpp
  src/linalg.cpp
  src/stats.cpp
  src/pp.cpp)

find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

target_include_directories(pptree
  PUBLIC include
  PUBLIC src)
  
target_link_libraries(pptree
  Eigen3::Eigen
  nlohmann_json::nlohmann_json)

# Test runner
enable_testing()

add_executable(pptree-test
  test/stats.test.cpp
  test/linalg.test.cpp
  test/pp.test.cpp
  test/pptree.test.cpp)

find_package(GTest REQUIRED)

target_link_libraries(pptree-test
  pptree
  gtest::gtest)

target_include_directories(pptree-test
  PUBLIC src)

include(GoogleTest)
gtest_discover_tests(pptree-test)



