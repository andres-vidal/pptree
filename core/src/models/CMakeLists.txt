add_library(ppforest2-models OBJECT
  Model.cpp
  Tree.cpp
  BootstrapTree.cpp
  Forest.cpp
  TreeNode.cpp
  TreeBranch.cpp
  TreeLeaf.cpp
  TrainingSpec.cpp
  strategies/pp/PDA.cpp
  strategies/vars/Uniform.cpp
  strategies/vars/All.cpp
  strategies/cutpoint/MeanOfMeans.cpp
  strategies/stop/PureNode.cpp
  strategies/binarize/LargestGap.cpp
  strategies/partition/ByGroup.cpp
  strategies/leaf/MajorityVote.cpp
  Projector.cpp
  VariableImportance.cpp
  Visualization.cpp)

target_link_libraries(ppforest2-models PRIVATE ppforest2-compile nlohmann_json::nlohmann_json)
