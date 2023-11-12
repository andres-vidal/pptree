cmake_minimum_required(VERSION 3.5)
 
project(pptree VERSION 0.0.0)

add_library(pptree
  src/pptree.cpp)

target_include_directories(pptree
  PUBLIC include)

# Executable (temporary)

add_executable(pptree-cli 
  main.cpp)

target_link_libraries(pptree-cli
  pptree)

target_include_directories(pptree-cli
  PUBLIC include)

