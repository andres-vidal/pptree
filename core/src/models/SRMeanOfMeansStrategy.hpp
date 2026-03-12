#pragma once

#include "models/SRStrategy.hpp"

namespace ppforest2::sr {
  /**
   * @brief Split threshold as the mean of two group means.
   *
   * Computes the midpoint between the projected means of the two
   * groups: (mean(group_1 * A) + mean(group_2 * A)) / 2.
   * This is the default rule used by PPforest.
   */
  struct SRMeanOfMeansStrategy : public SRStrategy {
    SRStrategy::Ptr clone() const override {
      return std::make_unique<SRMeanOfMeansStrategy>(*this);
    }

    types::Feature threshold(
      const types::FeatureMatrix& group_1,
      const types::FeatureMatrix& group_2,
      const pp::Projector&        projector) const override {
      return ((group_1 * projector).mean() + (group_2 * projector).mean()) / 2;
    }
  };

  /**
   * @brief Factory function for a mean-of-means split strategy.
   *
   * @return  Owned pointer to a SRMeanOfMeansStrategy.
   */
  inline SRStrategy::Ptr mean_of_means() {
    return std::make_unique<SRMeanOfMeansStrategy>();
  }
}
