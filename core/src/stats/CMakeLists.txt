add_library(pptree-stats OBJECT
  Stats.cpp
  GroupPartition.cpp
  ConfusionMatrix.cpp)

target_link_libraries(pptree-stats PRIVATE pptree-compile)
