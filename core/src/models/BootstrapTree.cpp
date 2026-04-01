#include "models/BootstrapTree.hpp"
#include "stats/Uniform.hpp"
#include <algorithm>
#include <numeric>
#include <set>
#include <Eigen/Dense>

using namespace ppforest2::stats;
using namespace ppforest2::types;

namespace ppforest2 {
  BootstrapTree::Ptr BootstrapTree::train(
      TrainingSpec::Ptr const& training_spec, FeatureMatrix const& x, GroupPartition const& group_spec, RNG& rng
  ) {
    std::vector<int> sample_indices;
    sample_indices.reserve(x.rows());

    for (auto const& group : group_spec.groups) {
      int const group_size = group_spec.group_size(group);
      int const min_index  = group_spec.group_start(group);
      int const max_index  = group_spec.group_end(group);

      Uniform const unif(min_index, max_index);

      for (int j = 0; j < group_size; j++) {
        sample_indices.push_back(unif(rng));
      }
    }

    std::sort(sample_indices.begin(), sample_indices.end());

    FeatureMatrix sampled_x = x(sample_indices, Eigen::all);

    Tree tree = Tree::train(*training_spec, sampled_x, group_spec, rng);

    return std::make_unique<BootstrapTree>(std::move(tree.root), training_spec, std::move(sample_indices));
  }

  std::vector<int> BootstrapTree::oob_indices(int n_total) const {
    std::set<int> in_bag(sample_indices.begin(), sample_indices.end());
    std::vector<int> oob;
    oob.reserve(static_cast<std::size_t>(n_total) - in_bag.size());

    for (int i = 0; i < n_total; ++i) {
      if (in_bag.find(i) == in_bag.end()) {
        oob.push_back(i);
      }
    }

    return oob;
  }

  ResponseVector BootstrapTree::predict_oob(FeatureMatrix const& x, std::vector<int> const& row_idx) const {
    if (row_idx.empty()) {
      return ResponseVector(0);
    }

    return predict(static_cast<FeatureMatrix>(x(row_idx, Eigen::all)));
  }

  BootstrapTree::BootstrapTree(TreeNode::Ptr root, TrainingSpec::Ptr spec, std::vector<int> samp)
      : Tree(std::move(root), std::move(spec))
      , sample_indices(std::move(samp)) {}
}
