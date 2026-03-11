#pragma once
#include <algorithm>
#include <numeric>

#include "models/DRStrategy.hpp"

namespace pptree::dr {
  /**
   * @brief No-op dimensionality reduction: selects all variables.
   *
   * Used with standard (non-random-forest) trees where all features
   * are available to the projection pursuit step at every node.
   */
  struct DRNoopStrategy : public DRStrategy {
    DRStrategy::Ptr clone() const override {
      return std::make_unique<DRNoopStrategy>(*this);
    }

    DRSpec select(
      types::FeatureMatrix const & x,
      stats::GroupPartition const& group_spec,
      stats::RNG &                 rng) const override {
      std::vector<int> all_indices(x.cols());
      std::iota(all_indices.begin(), all_indices.end(), 0);
      return DRSpec(all_indices, x.cols());
    }
  };

  /**
   * @brief Factory function for a no-op DR strategy.
   *
   * @return  Owned pointer to a DRNoopStrategy.
   */
  inline DRStrategy::Ptr noop() {
    return std::make_unique<DRNoopStrategy>();
  }
}
