#pragma once

#include "models/Tree.hpp"

namespace ppforest2 {
  /**
   * @brief A projection pursuit decision tree for regression.
   *
   * Leaves hold continuous mean response values produced by the
   * `MeanResponse` leaf strategy. Training requires a `y`
   * vector; in-place reordering of feature rows happens inside the
   * build loop via the `ByCutpoint` grouping strategy.
   *
   * `predict(data, Proportions)` is not meaningful for regression and
   * throws `std::invalid_argument`.
   */
  struct RegressionTree : public Tree {
    using Ptr = std::unique_ptr<RegressionTree>;

    RegressionTree(TreeNode::Ptr root, TrainingSpec::Ptr training_spec);

    // Bring base-class predict overloads into scope so name-hiding doesn't
    // drop them when we declare the Proportions override below.
    using Tree::predict;

    /**
     * @brief Train a regression tree with an external RNG.
     *
     * Takes `x` and `y` by **non-const reference**. The
     * `ByCutpoint` grouping strategy reorders rows in place on the
     * caller's storage — there is no internal copy. Callers must pass
     * buffers they own and are willing to see mutated. Typical callers:
     *
     * - **Bootstrap trees**: pass freshly-built local subsamples. Zero
     *   additional copies beyond the subsample itself.
     * - **Single-tree `Tree::train` path**: the top-level dispatcher
     *   holds the caller's data as `const&`, so it makes a single copy
     *   of `x` and `y` at the call site before invoking this
     *   function. The copy is visible at the caller, not hidden inside.
     *
     * @param training_spec  Training specification (must have `mode = Regression`).
     * @param x              Feature matrix (n × p), sorted by continuous response.
     *                       Will be permuted in place during training.
     * @param group_spec     Initial root group partition (typically a median split).
     * @param rng            Random number generator.
     * @param y   Continuous response vector (n), same order as `x`.
     *                       Will be permuted in place during training.
     */
    static Ptr train(
        TrainingSpec const& training_spec,
        types::FeatureMatrix& x,
        stats::GroupPartition const& group_spec,
        stats::RNG& rng,
        types::OutcomeVector& y
    );

    /** @brief Not supported for regression — throws. */
    types::FeatureMatrix predict(types::FeatureMatrix const& data, Proportions) const override;

    void accept(Model::Visitor& visitor) const override;
  };
}
