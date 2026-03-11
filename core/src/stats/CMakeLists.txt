add_library(pptree-stats OBJECT
  Stats.cpp
  Uniform.cpp
  Normal.cpp
  GroupPartition.cpp
  ConfusionMatrix.cpp
  Simulation.cpp)

target_link_libraries(pptree-stats PRIVATE pptree-compile)
