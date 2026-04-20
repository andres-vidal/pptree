#pragma once

#include "models/Tree.hpp"

namespace ppforest2 {
  /**
   * @brief A projection pursuit decision tree for classification.
   *
   * Leaves hold integer group labels produced by the configured leaf
   * strategy (default `MajorityVote`). The `predict(data, Proportions)`
   * overload returns a one-hot encoding of the predicted group.
   */
  struct ClassificationTree : public Tree {
    using Ptr = std::unique_ptr<ClassificationTree>;

    ClassificationTree(TreeNode::Ptr root, TrainingSpec::Ptr training_spec);

    // Bring base-class predict overloads into scope so name-hiding doesn't
    // drop them when we declare the Proportions override below.
    using Tree::predict;

    /**
     * @brief Train a classification tree with an external RNG.
     *
     * `x` and `y` are non-const for signature symmetry with
     * `RegressionTree::train` — classification doesn't mutate either.
     *
     * @param training_spec  Training specification (must have `mode = Classification`).
     * @param x              Feature matrix (n × p).
     * @param y              Response vector (integer class labels encoded as floats).
     * @param y_part     Initial root group partition.
     * @param rng            Random number generator.
     */
    static Ptr train(
        TrainingSpec const& training_spec,
        types::FeatureMatrix& x,
        types::OutcomeVector& y,
        stats::GroupPartition const& y_part,
        stats::RNG& rng
    );

    /** @brief One-hot encoding of the predicted group per row. */
    types::FeatureMatrix predict(types::FeatureMatrix const& data, Proportions) const override;

    void accept(Model::Visitor& visitor) const override;
  };
}
