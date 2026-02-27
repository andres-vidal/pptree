#include "BootstrapTree.hpp"
#include <algorithm>
#include <Eigen/Dense>

using namespace models::stats;
using namespace models::types;

models::BootstrapTree::Ptr models::BootstrapTree::train(
  TrainingSpec const&   training_spec,
  FeatureMatrix const&  x,
  GroupPartition const& group_spec,
  RNG &                 rng) {
  std::vector<int> sample_indices;
  sample_indices.reserve(x.rows());

  for (const auto& group : group_spec.groups) {
    const int group_size = group_spec.group_size(group);
    const int min_index  = group_spec.group_start(group);
    const int max_index  = group_spec.group_end(group);

    const Uniform unif(min_index, max_index);

    for (int j = 0; j < group_size; j++) {
      sample_indices.push_back(unif(rng));
    }
  }

  std::sort(sample_indices.begin(), sample_indices.end());

  FeatureMatrix sampled_x = x(sample_indices, Eigen::all);

  Tree tree = Tree::train(training_spec, sampled_x, group_spec, rng);

  return std::make_unique<BootstrapTree>(
    std::move(tree.root),
    training_spec.clone(),
    std::move(sample_indices)); // <-- important
}

models::BootstrapTree::BootstrapTree(
  TreeNode::Ptr     root,
  TrainingSpec::Ptr spec,
  std::vector<int>  samp)
  : Tree(std::move(root), std::move(spec)),
  sample_indices(std::move(samp)) {
}
