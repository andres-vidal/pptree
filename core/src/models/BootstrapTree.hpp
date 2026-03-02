#pragma once

#include "utils/Types.hpp"
#include "models/Tree.hpp"

namespace pptree {
  struct BootstrapTree : public Tree {
    using Tree::Tree;
    using Ptr = std::unique_ptr<BootstrapTree>;

    static Ptr train(
      TrainingSpec const&          training_spec,
      types::FeatureMatrix const&  x,
      stats::GroupPartition const& group_spec,
      stats::RNG &                 rng);

    std::vector<int> sample_indices;

    BootstrapTree(TreeNode::Ptr root, TrainingSpec::Ptr spec, std::vector<int> samp);
  };
}
