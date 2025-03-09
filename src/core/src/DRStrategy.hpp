#pragma once
#include <algorithm>

#include "Data.hpp"
#include "Uniform.hpp"
#include "Invariant.hpp"

namespace models::dr::strategy {
  template<typename T>
  struct DRStrategy {
    virtual ~DRStrategy() = default;
    virtual std::unique_ptr<DRStrategy<T> > clone() const = 0;

    virtual stats::Data<T> select_variables(
      const stats::Data<T>& data) const = 0;

    stats::Data<T> operator()(
      const stats::Data<T>& data) const {
      return select_variables(data);
    }
  };

  template<typename T>
  struct ReduceNoneStrategy : public DRStrategy<T> {
    std::unique_ptr<DRStrategy<T> > clone() const override {
      return std::make_unique<ReduceNoneStrategy<T> >(*this);
    }

    stats::Data<T> select_variables(
      const stats::Data<T>& data) const override {
      return data;
    }
  };

  template<typename T>
  struct ReduceUniformlyStrategy : public DRStrategy<T> {
    const int n_vars;

    explicit ReduceUniformlyStrategy(const int n_vars) : n_vars(n_vars) {
      invariant(n_vars > 0, "The number of variables must be greater than 0.");
    }

    std::unique_ptr<DRStrategy<T> > clone() const override {
      return std::make_unique<ReduceUniformlyStrategy<T> >(*this);
    }

    stats::Data<T> select_variables(
      const stats::Data<T>& data) const override {
      invariant(n_vars <= data.cols(), "The number of variables must be less than or equal to the number of columns in the data.");

      if (n_vars == data.cols()) return data;

      LOG_INFO << "Selecting " << n_vars << " variables uniformly." << std::endl;

      stats::Uniform unif(0, data.cols() - 1);

      std::vector<int> var_sampled_indices = unif.distinct(n_vars);

      LOG_INFO << "Selected variables: " << var_sampled_indices << std::endl;

      return data(Eigen::all, var_sampled_indices);
    }
  };



  template<typename T>
  std::unique_ptr<DRStrategy<T> > all() {
    return std::make_unique<ReduceNoneStrategy<T> >();
  }

  template<typename T>
  std::unique_ptr<DRStrategy<T> > uniform(const int n_vars) {
    return std::make_unique<ReduceUniformlyStrategy<T> >(n_vars);
  }
}
