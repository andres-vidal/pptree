#pragma once
#include <algorithm>

#include "DRStrategy.hpp"
#include "Uniform.hpp"
#include "Invariant.hpp"

namespace models::dr::strategy {
  struct DRUniformStrategy : public DRStrategy {
    const int n_vars;

    explicit DRUniformStrategy(const int n_vars) : n_vars(n_vars) {
      invariant(n_vars > 0, "The number of variables must be greater than 0.");
    }

    DRStrategy::Ptr clone() const override {
      return std::make_unique<DRUniformStrategy>(*this);
    }

    DRSpec select(
      types::FeatureMatrix const &  x,
      stats::GroupPartition const & group_spec,
      stats::RNG &                  rng) const override {
      invariant(n_vars <= x.cols(), "The number of variables must be less than or equal to the number of columns in the data.");

      if (n_vars == x.cols()) {
        std::vector<int> all_indices(x.cols());
        std::iota(all_indices.begin(), all_indices.end(), 0);

        return DRSpec(all_indices, x.cols());
      }

      stats::Uniform unif(0, x.cols() - 1);
      std::vector<int> selected_indices = unif.distinct(n_vars, rng);

      return DRSpec(selected_indices, x.cols());
    }
  };


  inline DRStrategy::Ptr uniform(const int n_vars) {
    return std::make_unique<DRUniformStrategy>(n_vars);
  }
}
