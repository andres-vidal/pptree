#pragma once

#include "DataSpec.hpp"
#include "SortedDataSpec.hpp"

namespace models::stats {
  template<typename T, typename G>
  struct ReducedDataSpec : public SortedDataSpec<T, G> {
    private:
      const size_t original_columns;
      const std::vector<int> selected_indices;

      ReducedDataSpec(const SortedDataSpec<T, G>& data, const std::vector<int>& indices, const size_t original_columns) :
        SortedDataSpec<T, G>(data),
        original_columns(original_columns),
        selected_indices(indices) {
      }

    public:

      ReducedDataSpec(const SortedDataSpec<T, G>& data, const std::vector<int>& indices) :
        SortedDataSpec<T, G>(data.analog(data.x(Eigen::all, indices))),
        original_columns(data.x.cols()),
        selected_indices(indices) {
      }

      DataColumn<T> expand(const DataColumn<T>& reduced_vector) const {
        LOG_INFO << "Expanding reduced vector:" << reduced_vector.transpose() << std::endl;
        LOG_INFO << "Full vector size:" << original_columns << std::endl;
        LOG_INFO << "Selected indices:" << selected_indices << std::endl;

        invariant(reduced_vector.size() == selected_indices.size(), "Reduced vector size must match number of selected variables");

        DataColumn<T> full_vector = DataColumn<T>::Zero(original_columns);
        for (int i = 0; i < selected_indices.size(); ++i) {
          full_vector(selected_indices[i]) = reduced_vector(i);
        }

        LOG_INFO << "Expanded vector:" << full_vector.transpose() << std::endl;

        return full_vector;
      }

      ReducedDataSpec<T, G> remap(const std::map<G, int>& mapping) const {
        return ReducedDataSpec<T, G>(
          SortedDataSpec<T, G>::remap(mapping),
          selected_indices,
          original_columns);
      }
  };
}
