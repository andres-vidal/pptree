#pragma once
#include <algorithm>

#include "models/DRStrategy.hpp"

namespace pptree::dr {
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

  inline DRStrategy::Ptr noop() {
    return std::make_unique<DRNoopStrategy>();
  }
}
