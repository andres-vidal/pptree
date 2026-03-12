add_library(ppforest2-models OBJECT
  Tree.cpp
  BootstrapTree.cpp
  Forest.cpp
  TreeNode.cpp
  TreeCondition.cpp
  TreeResponse.cpp
  PPGLDAStrategy.cpp
  Projector.cpp
  VariableImportance.cpp
  Visualization.cpp)

target_link_libraries(ppforest2-models PRIVATE ppforest2-compile)
