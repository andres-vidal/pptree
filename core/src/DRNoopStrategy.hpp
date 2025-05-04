#pragma once
#include <algorithm>

#include "DRStrategy.hpp"

namespace models::dr::strategy {
  template<typename T, typename G>
  struct DRNoopStrategy : public DRStrategy<T, G> {
    DRStrategyPtr<T, G> clone() const override {
      return std::make_unique<DRNoopStrategy<T, G> >(*this);
    }

    DRSpec<T, G> select(
      const stats::GroupSpec<T, G>& spec) const override {
      std::vector<int> all_indices(spec.cols());
      std::iota(all_indices.begin(), all_indices.end(), 0);
      return DRSpec<T, G>(all_indices, spec.cols());
    }
  };

  template<typename T, typename G>
  DRStrategyPtr<T, G> noop() {
    return std::make_unique<DRNoopStrategy<T, G> >();
  }
}
