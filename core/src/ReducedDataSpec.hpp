#pragma once

#include "DataSpec.hpp"
#include "SortedDataSpec.hpp"

namespace models::stats {
  template<typename T, typename G>
  struct ReducedDataSpec : public SortedDataSpec<T, G> {
    private:
      const std::vector<int> selected_indices;
      const SortedDataSpec<T, G> reduced_spec;

    public:
      ReducedDataSpec(
        const SortedDataSpec<T, G>& data,
        const std::vector<int>&     indices)
        : SortedDataSpec<T, G>(data.x, data.y, data.classes),
        selected_indices(indices),
        reduced_spec(data.analog(data.x(Eigen::all, indices))) {
      }

      Data<T> bgss() const override {
        return reduced_spec.bgss();
      }

      Data<T> wgss() const override {
        return reduced_spec.wgss();
      }

      DataColumn<T> expand(const DataColumn<T>& reduced_vector) const {
        LOG_INFO << "Expanding reduced vector:" << reduced_vector.transpose() << std::endl;
        LOG_INFO << "Full vector size:" << this->x.cols() << std::endl;
        LOG_INFO << "Selected indices:" << selected_indices << std::endl;

        invariant(reduced_vector.size() == selected_indices.size(), "Reduced vector size must match number of selected variables");

        DataColumn<T> full_vector = DataColumn<T>::Zero(this->x.cols());
        for (int i = 0; i < selected_indices.size(); ++i) {
          full_vector(selected_indices[i]) = reduced_vector(i);
        }

        LOG_INFO << "Expanded vector:" << full_vector.transpose() << std::endl;

        return full_vector;
      }

      ReducedDataSpec<T, G> remap(const std::map<G, int>& mapping) const {
        return ReducedDataSpec<T, G>(
          SortedDataSpec<T, G>::remap(mapping),
          selected_indices);
      }

      Data<T> reduced_x() const {
        return reduced_spec.x;
      }
  };
}
