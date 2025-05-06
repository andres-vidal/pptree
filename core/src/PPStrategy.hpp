#pragma once

#include "Projector.hpp"
#include "DataSpec.hpp"
#include <set>
#include <vector>

namespace models::pp::strategy {
  template<typename T, typename G>
  struct PPStrategy;

  template<typename T, typename G>
  using PPStrategyPtr = std::unique_ptr<PPStrategy<T, G> >;

  template<typename T, typename G>
  struct PPStrategy {
    virtual ~PPStrategy()                     = default;
    virtual PPStrategyPtr<T, G> clone() const = 0;

    virtual T index(
      const stats::DataSpec<T, G>& spec,
      const Projector<T>&          projector) const = 0;

    virtual Projector<T> optimize(
      const stats::DataSpec<T, G>& spec) const = 0;

    Projector<T> operator()(const stats::DataSpec<T, G>& spec) const {
      return optimize(spec);
    }
  };
}
