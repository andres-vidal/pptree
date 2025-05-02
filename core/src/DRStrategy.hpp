#pragma once
#include <algorithm>

#include "Data.hpp"
#include "Invariant.hpp"

namespace models::dr::strategy {
  template<typename T, typename G>
  struct DRStrategy;
  template<typename T, typename G>
  using DRStrategyPtr = std::unique_ptr<DRStrategy<T, G> >;

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
    virtual ~DRStrategy()                     = default;
    virtual DRStrategyPtr<T, G> clone() const = 0;

    virtual DRSpec<T, G> reduce(const stats::SortedDataSpec<T, G>& data) const = 0;

    DRSpec<T, G> operator()(const stats::SortedDataSpec<T, G>& data) const {
      return reduce(data);
    }
  };
}
