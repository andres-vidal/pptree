add_library(pptree-utils OBJECT
  Invariant.cpp)

target_link_libraries(pptree-utils PRIVATE pptree-compile)
