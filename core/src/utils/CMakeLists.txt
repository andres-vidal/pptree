add_library(ppforest2-utils OBJECT
  Invariant.cpp)

target_link_libraries(ppforest2-utils PRIVATE ppforest2-compile)
