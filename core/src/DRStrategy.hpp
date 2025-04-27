#pragma once
#include <algorithm>

#include "Data.hpp"
#include "Uniform.hpp"
#include "Invariant.hpp"
#include "ReducedDataSpec.hpp"

namespace models::dr::strategy {
  template<typename T, typename G>
  struct DRStrategy {
    virtual ~DRStrategy()                                    = default;
    virtual std::unique_ptr<DRStrategy<T, G> > clone() const = 0;

    virtual stats::ReducedDataSpec<T, G> reduce(
      const stats::SortedDataSpec<T, G>& data) const = 0;

    stats::ReducedDataSpec<T, G> operator()(
      const stats::SortedDataSpec<T, G>& data) const {
      return reduce(data);
    }
  };

  template<typename T, typename G>
  struct ReduceNoneStrategy : public DRStrategy<T, G> {
    std::unique_ptr<DRStrategy<T, G> > clone() const override {
      return std::make_unique<ReduceNoneStrategy<T, G> >(*this);
    }

    stats::ReducedDataSpec<T, G> reduce(
      const stats::SortedDataSpec<T, G>& data) const override {
      std::vector<int> all_indices(data.x.cols());
      std::iota(all_indices.begin(), all_indices.end(), 0);
      return stats::ReducedDataSpec<T, G>(data, all_indices);
    }
  };

  template<typename T, typename G>
  struct ReduceUniformlyStrategy : public DRStrategy<T, G> {
    const int n_vars;

    explicit ReduceUniformlyStrategy(const int n_vars) : n_vars(n_vars) {
      invariant(n_vars > 0, "The number of variables must be greater than 0.");
    }

    std::unique_ptr<DRStrategy<T, G> > clone() const override {
      return std::make_unique<ReduceUniformlyStrategy<T, G> >(*this);
    }

    stats::ReducedDataSpec<T, G> reduce(
      const stats::SortedDataSpec<T, G>& data) const override {
      invariant(n_vars <= data.x.cols(), "The number of variables must be less than or equal to the number of columns in the data.");

      if (n_vars == data.x.cols()) {
        std::vector<int> all_indices(data.x.cols());
        std::iota(all_indices.begin(), all_indices.end(), 0);
        return stats::ReducedDataSpec<T, G>(data, all_indices);
      }

      LOG_INFO << "Selecting " << n_vars << " variables uniformly." << std::endl;

      stats::Uniform unif(0, data.x.cols() - 1);
      std::vector<int> selected_indices = unif.distinct(n_vars);

      LOG_INFO << "Selected variables: " << selected_indices << std::endl;

      return stats::ReducedDataSpec<T, G>(data, selected_indices);
    }
  };

  template<typename T, typename G>
  std::unique_ptr<DRStrategy<T, G> > all() {
    return std::make_unique<ReduceNoneStrategy<T, G> >();
  }

  template<typename T, typename G>
  std::unique_ptr<DRStrategy<T, G> > uniform(const int n_vars) {
    return std::make_unique<ReduceUniformlyStrategy<T, G> >(n_vars);
  }
}
