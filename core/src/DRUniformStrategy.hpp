#pragma once
#include <algorithm>

#include "DRStrategy.hpp"
#include "Uniform.hpp"
#include "Invariant.hpp"

namespace models::dr::strategy {
  template<typename T, typename G>
  struct DRUniformStrategy : public DRStrategy<T, G> {
    const int n_vars;

    explicit DRUniformStrategy(const int n_vars) : n_vars(n_vars) {
      invariant(n_vars > 0, "The number of variables must be greater than 0.");
    }

    DRStrategyPtr<T, G> clone() const override {
      return std::make_unique<DRUniformStrategy<T, G> >(*this);
    }

    DRSpec<T, G> select(
      const stats::Data<T> &     x,
      const stats::GroupSpec<G>& group_spec,
      stats::RNG &               rng) const override {
      invariant(n_vars <= x.cols(), "The number of variables must be less than or equal to the number of columns in the data.");

      if (n_vars == x.cols()) {
        std::vector<int> all_indices(x.cols());
        std::iota(all_indices.begin(), all_indices.end(), 0);

        return DRSpec<T, G>(all_indices, x.cols());
      }

      stats::Uniform unif(0, x.cols() - 1);
      std::vector<int> selected_indices = unif.distinct(n_vars, rng);

      return DRSpec<T, G>(selected_indices, x.cols());
    }
  };


  template<typename T, typename G>
  DRStrategyPtr<T, G> uniform(const int n_vars) {
    return std::make_unique<DRUniformStrategy<T, G> >(n_vars);
  }
}
