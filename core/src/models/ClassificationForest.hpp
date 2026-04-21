#pragma once

#include "models/Forest.hpp"

#include <optional>

namespace ppforest2 {
  /**
   * @brief Random forest of classification trees.
   *
   * Aggregates per-tree predictions by majority vote. OOB predictions
   * use the same majority vote with `-1` as the "no OOB tree" sentinel.
   * Supports `predict(data, Proportions)` returning per-group vote
   * proportions.
   */
  struct ClassificationForest : public Forest {
    using Ptr = std::unique_ptr<ClassificationForest>;

    ClassificationForest();
    explicit ClassificationForest(TrainingSpec::Ptr training_spec);

    // Bring the Forest::predict(FeatureMatrix) overload into scope.
    using Forest::predict;

    static Ptr train(
        TrainingSpec const& training_spec,
        types::FeatureMatrix const& x,
        types::OutcomeVector const& y
    );

    types::Outcome predict(types::FeatureVector const& data) const override;
    types::FeatureMatrix predict(types::FeatureMatrix const& data, Proportions) const override;

    types::OutcomeVector oob_predict(types::FeatureMatrix const& x) const override;

    /**
     * @brief Out-of-bag misclassification rate.
     *
     * Truth vector carries integer class labels as `OutcomeVector`; the
     * method casts locally to `GroupIdVector` for comparison against OOB
     * votes. Observations with no OOB tree (value `-1` in `oob_predict`)
     * are excluded.
     *
     * Returns `std::nullopt` if no observation has any OOB tree — a
     * well-defined "no OOB data" signal rather than a magic sentinel.
     * R callers see this as `NA_real_`; CLI callers see "not available"
     * instead of a numeric value. Upstream of this refactor the method
     * returned `-1.0` for the same condition, which was not
     * distinguishable from a real (albeit impossible) error rate.
     */
    std::optional<double>
    oob_error(types::FeatureMatrix const& x, types::OutcomeVector const& y) const;

    /** @brief Convenience overload accepting integer labels. */
    std::optional<double>
    oob_error(types::FeatureMatrix const& x, types::GroupIdVector const& y) const {
      return oob_error(x, types::OutcomeVector(y.cast<types::Outcome>()));
    }

    // VI — accepts y as OutcomeVector for the base signature; internally
    // casts to GroupIdVector since classification measures work on labels.
    types::FeatureVector vi_permuted(
        types::FeatureMatrix const& x,
        types::OutcomeVector const& y,
        int seed
    ) const override;

    types::FeatureVector vi_weighted_projections(
        types::FeatureMatrix const& x,
        types::OutcomeVector const& y,
        types::FeatureVector const* scale = nullptr
    ) const override;

    void accept(Model::Visitor& visitor) const override;
  };
}
