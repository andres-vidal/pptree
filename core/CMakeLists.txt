cmake_minimum_required(VERSION 3.5)
 
project(pptree VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(pptree
  src/pptree.cpp)

target_include_directories(pptree
  PUBLIC include)


# Test runner
# enable_testing()

# add_executable(pptree-test)

# find_package(GTest REQUIRED)

# target_link_libraries(pptree-test
#   pptree
#   gtest::gtest)

# target_include_directories(pptree-test
#   PUBLIC src)

# include(GoogleTest)
# gtest_discover_tests(pptree-test)

