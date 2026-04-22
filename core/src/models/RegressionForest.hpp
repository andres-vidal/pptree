#pragma once

#include "models/Forest.hpp"

#include <optional>

namespace ppforest2 {
  /**
   * @brief Random forest of regression trees.
   *
   * Aggregates per-tree predictions by arithmetic mean. OOB predictions
   * use the same mean with `NaN` as the "no OOB tree" sentinel (valid
   * regression predictions can be any finite value, so NaN is the only
   * value that cannot collide with a real prediction).
   *
   * `predict(data, Proportions)` is not meaningful for regression and
   * throws.
   */
  struct RegressionForest : public Forest {
    using Ptr = std::unique_ptr<RegressionForest>;

    RegressionForest();
    explicit RegressionForest(TrainingSpec::Ptr training_spec);

    // Bring the Forest::predict(FeatureMatrix) overload into scope.
    using Forest::predict;

    static Ptr train(TrainingSpec const& training_spec, types::FeatureMatrix const& x, types::OutcomeVector const& y);

    types::Outcome predict(types::FeatureVector const& data) const override;
    types::FeatureMatrix predict(types::FeatureMatrix const& data, Proportions) const override;

    types::OutcomeVector oob_predict(types::FeatureMatrix const& x) const override;

    /**
     * @brief Out-of-bag mean squared error.
     *
     * Truth vector is the continuous response (`OutcomeVector`). Rows with
     * no OOB tree (NaN in `oob_predict`) are excluded from the sum.
     *
     * Returns `std::nullopt` when no observation has any OOB tree — a
     * well-defined "no OOB data" signal rather than a magic sentinel.
     * R callers see this as `NA_real_`; CLI callers see "not available".
     */
    std::optional<double> oob_error(types::FeatureMatrix const& x, types::OutcomeVector const& y) const;

    /**
     * @brief VI1 — per-variable NMSE increase on permuted OOB rows.
     *
     * For each tree, computes baseline OOB MSE and per-column permuted
     * OOB MSE, then accumulates `(perm_mse - baseline_mse) / Var(y_oob)`.
     * Dividing by `Var(y_oob)` yields a scale-free signal analogous to
     * the accuracy drop used for classification. Trees whose OOB
     * response has zero variance contribute nothing.
     */
    types::FeatureVector
    vi_permuted(types::FeatureMatrix const& x, types::OutcomeVector const& y, int seed) const override;

    /**
     * @brief VI3 — weighted projections with NMSE-based quality weight.
     *
     * Each tree's contribution is weighted by `max(0, 1 - NMSE_k)` where
     * `NMSE_k = MSE_k / Var(y_oob_k)`. The classification `(G - 1)`
     * denominator is dropped — regression trees have no natural maximum
     * split count to normalise against. Denominator is `valid_trees`.
     */
    types::FeatureVector vi_weighted_projections(
        types::FeatureMatrix const& x, types::OutcomeVector const& y, types::FeatureVector const* scale = nullptr
    ) const override;

    void accept(Model::Visitor& visitor) const override;
  };
}
