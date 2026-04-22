#include "models/RegressionTree.hpp"

#include "utils/Invariant.hpp"

#include <stdexcept>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2 {
  RegressionTree::RegressionTree(TreeNode::Ptr root, TrainingSpec::Ptr training_spec)
      : Tree(std::move(root), std::move(training_spec)) {
    // See the mirror check in ClassificationTree: mode/class mismatches
    // silently corrupt downstream math, so fail loudly at construction.
    // Tests that only care about tree structure should use
    // `test::regression_spec()`.
    invariant(
        this->training_spec && this->training_spec->mode == types::Mode::Regression,
        "RegressionTree requires a non-null TrainingSpec with mode = Regression"
    );
  }

  RegressionTree::Ptr RegressionTree::train(
      TrainingSpec const& training_spec,
      FeatureMatrix& x,
      OutcomeVector& y,
      GroupPartition const& y_part,
      stats::RNG& rng
  ) {
    invariant(training_spec.mode == Mode::Regression, "RegressionTree::train requires mode = Regression");

    // ByCutpoint reorders rows of `x` and `y` in place on the caller's
    // storage — no copy. Caller is responsible for providing a buffer
    // it is willing to see mutated.
    TreeNode::Ptr root_ptr = Tree::build_root(training_spec, x, y, y_part, rng);

    return std::make_unique<RegressionTree>(std::move(root_ptr), TrainingSpec::make(training_spec));
  }

  FeatureMatrix RegressionTree::predict(FeatureMatrix const& /*data*/, Proportions) const {
    throw std::invalid_argument("Vote proportions are not available for regression trees."
                                "Use predict(data) for numeric predictions.");
  }

  void RegressionTree::accept(Model::Visitor& visitor) const {
    visitor.visit(*this);
  }
}
