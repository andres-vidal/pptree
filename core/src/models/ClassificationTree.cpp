#include "models/ClassificationTree.hpp"

#include "utils/Invariant.hpp"

#include <map>
#include <set>
#include <vector>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2 {
  ClassificationTree::ClassificationTree(TreeNode::Ptr root, TrainingSpec::Ptr training_spec)
      : Tree(std::move(root), std::move(training_spec)) {
    // Guard against a caller wrapping a classification tree over a
    // regression spec or a null spec. `predict` / VI / aggregation all
    // assume the mode matches the concrete subclass, and a mismatch
    // produces silent garbage rather than a clear error. Tests that only
    // care about tree structure should use `test::classification_spec()`.
    invariant(
        this->training_spec && this->training_spec->mode == types::Mode::Classification,
        "ClassificationTree requires a non-null TrainingSpec with mode = Classification"
    );
  }

  ClassificationTree::Ptr ClassificationTree::train(
      TrainingSpec const& training_spec,
      FeatureMatrix const& x,
      GroupPartition const& group_spec,
      stats::RNG& rng
  ) {
    invariant(
        training_spec.mode == Mode::Classification,
        "ClassificationTree::train requires mode = Classification"
    );

    TreeNode::Ptr root_ptr = Tree::build_root(training_spec, x, group_spec, rng);

    return std::make_unique<ClassificationTree>(std::move(root_ptr), TrainingSpec::make(training_spec));
  }

  FeatureMatrix ClassificationTree::predict(FeatureMatrix const& data, Proportions) const {
    std::set<GroupId> group_set = root->node_groups();
    std::vector<GroupId> groups(group_set.begin(), group_set.end());
    int const G = static_cast<int>(groups.size());

    std::map<GroupId, int> group_to_col;
    for (int g = 0; g < G; ++g) {
      group_to_col[groups[static_cast<std::size_t>(g)]] = g;
    }

    int const n               = static_cast<int>(data.rows());
    FeatureMatrix proportions = FeatureMatrix::Zero(n, G);

    for (int i = 0; i < n; ++i) {
      Outcome const pred = Tree::predict(static_cast<FeatureVector>(data.row(i)));
      proportions(i, group_to_col[static_cast<GroupId>(pred)]) = Feature(1);
    }

    return proportions;
  }

  void ClassificationTree::accept(Model::Visitor& visitor) const {
    visitor.visit(*this);
  }
}
