#pragma once

#include "models/SRStrategy.hpp"
#include "models/Strategy.hpp"
#include "utils/JsonValidation.hpp"

namespace ppforest2::sr {
  /**
   * @brief Split threshold as the mean of two group means.
   *
   * Computes the midpoint between the projected means of the two
   * groups: (mean(group_1 * A) + mean(group_2 * A)) / 2.
   * This is the default rule used by PPforest.
   */
  struct SRMeanOfMeansStrategy : public SRStrategy {
    void to_json(nlohmann::json& j) const override;
    std::string display_name() const override { return "Mean of means"; }

    types::Feature threshold(
        types::FeatureMatrix const& group_1, types::FeatureMatrix const& group_2, pp::Projector const& projector
    ) const override;

    static SRStrategy::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(SRStrategy, "mean_of_means")
  };

  /** @brief Factory function for a mean-of-means split strategy. */
  SRStrategy::Ptr mean_of_means();
}
