add_library(pptree-models OBJECT
  Tree.cpp
  BootstrapTree.cpp
  Forest.cpp
  TreeNode.cpp
  TreeCondition.cpp
  TreeResponse.cpp
  PPGLDAStrategy.cpp
  Projector.cpp
  VariableImportance.cpp)

target_link_libraries(pptree-models PRIVATE pptree-compile)
