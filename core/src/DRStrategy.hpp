#pragma once
#include <algorithm>

#include "Data.hpp"
#include "Uniform.hpp"
#include "Invariant.hpp"

namespace models::dr::strategy {
  template<typename T, typename G>
  struct DRSpec {
    const std::vector<int> selected_cols;
    const size_t original_size;

    DRSpec(const std::vector<int>& selected_cols, const size_t original_size) :
      selected_cols(selected_cols),
      original_size(original_size) {
    }

    stats::DataColumn<T> expand(const stats::DataColumn<T>& reduced_vector) const {
      LOG_INFO << "Expanding reduced vector:" << reduced_vector.transpose() << std::endl;
      LOG_INFO << "Full vector size:" << original_size << std::endl;
      LOG_INFO << "Selected indices:" << selected_cols << std::endl;

      invariant(reduced_vector.size() == selected_cols.size(), "Reduced vector size must match number of selected variables");

      stats::DataColumn<T> full_vector = stats::DataColumn<T>::Zero(original_size);

      for (int i = 0; i < selected_cols.size(); ++i) {
        full_vector(selected_cols[i]) = reduced_vector(i);
      }

      LOG_INFO << "Expanded vector:" << full_vector.transpose() << std::endl;

      return full_vector;
    }

    stats::SortedDataSpec<T, G> reduce(const stats::SortedDataSpec<T, G>& data) const {
      return data.analog(data.x(Eigen::all, selected_cols));
    }
  };

  template<typename T, typename G>
  struct DRStrategy {
    virtual ~DRStrategy()                                    = default;
    virtual std::unique_ptr<DRStrategy<T, G> > clone() const = 0;

    virtual DRSpec<T, G> reduce(const stats::SortedDataSpec<T, G>& data) const = 0;

    DRSpec<T, G> operator()(const stats::SortedDataSpec<T, G>& data) const {
      return reduce(data);
    }
  };

  template<typename T, typename G>
  struct ReduceNoneStrategy : public DRStrategy<T, G> {
    std::unique_ptr<DRStrategy<T, G> > clone() const override {
      return std::make_unique<ReduceNoneStrategy<T, G> >(*this);
    }

    DRSpec<T, G> reduce(
      const stats::SortedDataSpec<T, G>& data) const override {
      std::vector<int> all_indices(data.x.cols());
      std::iota(all_indices.begin(), all_indices.end(), 0);
      return DRSpec<T, G>(all_indices, data.x.cols());
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

    DRSpec<T, G> reduce(
      const stats::SortedDataSpec<T, G>& data) const override {
      invariant(n_vars <= data.x.cols(), "The number of variables must be less than or equal to the number of columns in the data.");

      if (n_vars == data.x.cols()) {
        std::vector<int> all_indices(data.x.cols());
        std::iota(all_indices.begin(), all_indices.end(), 0);

        return DRSpec<T, G>(all_indices, data.x.cols());
      }

      LOG_INFO << "Selecting " << n_vars << " variables uniformly." << std::endl;

      stats::Uniform unif(0, data.x.cols() - 1);
      std::vector<int> selected_indices = unif.distinct(n_vars);

      LOG_INFO << "Selected variables: " << selected_indices << std::endl;

      return DRSpec<T, G>(selected_indices, data.x.cols());
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
