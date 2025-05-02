#pragma once
#include <algorithm>

#include "DRStrategy.hpp"

namespace models::dr::strategy {
  template<typename T, typename G>
  struct DRNoopStrategy : public DRStrategy<T, G> {
    DRStrategyPtr<T, G> clone() const override {
      return std::make_unique<DRNoopStrategy<T, G> >(*this);
    }

    DRSpec<T, G> reduce(
      const stats::SortedDataSpec<T, G>& data) const override {
      std::vector<int> all_indices(data.x.cols());
      std::iota(all_indices.begin(), all_indices.end(), 0);
      return DRSpec<T, G>(all_indices, data.x.cols());
    }
  };

  template<typename T, typename G>
  DRStrategyPtr<T, G> noop() {
    return std::make_unique<DRNoopStrategy<T, G> >();
  }
}
