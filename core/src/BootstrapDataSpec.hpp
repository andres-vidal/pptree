#pragma once

#include "SortedDataSpec.hpp"
#include "Uniform.hpp"
#include "Invariant.hpp"

#include <algorithm>

namespace models::stats {
  template<typename T, typename G>
  struct BootstrapDataSpec : SortedDataSpec<T, G> {
    const std::vector<int> sample_indices;
    const std::set<int> oob_indices;

    BootstrapDataSpec(
      const Data<T> &         x,
      const DataColumn<G> &   y,
      const std::set<G> &     classes,
      const std::vector<int> &sample_indices)
      : SortedDataSpec<T, G>(x, y, classes),
      sample_indices(sample_indices),
      oob_indices(init_oob_indices(x, sample_indices)) {
    }

    BootstrapDataSpec(
      const Data<T> &         x,
      const DataColumn<G> &   y,
      const std::vector<int> &sample_indices)
      : BootstrapDataSpec<T, G>(x, y, unique(y), sample_indices) {
    }

    SortedDataSpec<T, G> get_sample() const {
      return SortedDataSpec<T, G>(
        this->x(this->sample_indices, Eigen::all),
        this->y(this->sample_indices, Eigen::all));
    }

    SortedDataSpec<T, G> get_oob() const {
      std::vector<int> oob_indices_vec(oob_indices.begin(), oob_indices.end());

      return SortedDataSpec<T, G>(
        this->x(oob_indices_vec, Eigen::all),
        this->y(oob_indices_vec, Eigen::all));
    }

    std::tuple<Data<T>, DataColumn<G>, std::set<G> > unwrap() const override {
      return this->get_sample().unwrap();
    }

    private:

      static std::set<int> init_oob_indices(const Data<T> &data, const std::vector<int> &sample_indices) {
        std::set<int> all_indices;

        for (int i = 0; i < data.rows(); i++) {
          all_indices.insert(i);
        }

        std::set<int> iob_indices(sample_indices.begin(), sample_indices.end());
        std::set<int> oob_indices;
        std::set_difference(
          all_indices.begin(),
          all_indices.end(),
          iob_indices.begin(),
          iob_indices.end(),
          std::inserter(oob_indices, oob_indices.end()));

        return oob_indices;
      }
  };

  template<typename T, typename G>
  BootstrapDataSpec<T, G> stratified_proportional_sample(
    const SortedDataSpec<T, G> &data,
    const int                   size) {
    invariant(size > 0, "Sample size must be greater than 0.");
    invariant(size <= data.y.rows(), "Sample size cannot be larger than the number of rows in the data.");

    const int data_size = data.y.rows();

    std::vector<int> sample_indices;
    sample_indices.reserve(size);

    for (const G& group : data.classes) {
      const int group_size = data.group_size(group);
      const int group_sample_size = std::max(1, (int)std::round(group_size / (float)data_size * size));

      for (int i = 0; i < group_sample_size; i++) {
        const Uniform unif(data.group_start(group), data.group_end(group));
        sample_indices.push_back(unif());
      }
    }

    return BootstrapDataSpec<T, G>(data.x, data.y, data.classes, sample_indices);
  }

  template<typename T, typename G>
  BootstrapDataSpec<T, G> center(const BootstrapDataSpec<T, G> &data) {
    return BootstrapDataSpec<T, G>(center(data.x), data.y, data.classes, data.sample_indices);
  }

  template<typename T, typename G>
  BootstrapDataSpec<T, G> descale(const BootstrapDataSpec<T, G> &data) {
    return BootstrapDataSpec<T, G>(descale(data.x), data.y, data.classes, data.sample_indices);
  }
}
