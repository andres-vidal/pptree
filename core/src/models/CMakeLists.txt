add_library(ppforest2-models OBJECT
  Model.cpp
  Tree.cpp
  BootstrapTree.cpp
  Forest.cpp
  TreeNode.cpp
  TreeCondition.cpp
  TreeResponse.cpp
  TrainingSpec.cpp
  PPPDAStrategy.cpp
  DRUniformStrategy.cpp
  DRNoopStrategy.cpp
  SRMeanOfMeansStrategy.cpp
  Projector.cpp
  VariableImportance.cpp
  Visualization.cpp)

target_link_libraries(ppforest2-models PRIVATE ppforest2-compile nlohmann_json::nlohmann_json)
