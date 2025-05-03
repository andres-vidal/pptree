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

    BootstrapDataSpec() :
      SortedDataSpec<T, G>(),
      sample_indices(),
      oob_indices() {
    }

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
      return SortedDataSpec<T, G>::select(sample_indices);
    }

    SortedDataSpec<T, G> get() const {
      return get_sample();
    }

    SortedDataSpec<T, G> get_oob() const {
      return SortedDataSpec<T, G>::select(oob_indices);
    }

    private:

      static std::set<int> init_oob_indices(const Data<T> &data, const std::vector<int> &sample_indices) {
        std::set<int> oob_indices;

        std::set<int> iob_indices(sample_indices.begin(), sample_indices.end());

        for (int i = 0; i < data.rows(); i++) {
          if (iob_indices.count(i) == 0) oob_indices.insert(i);
        }

        return oob_indices;
      }
  };

  template<typename T, typename G>
  BootstrapDataSpec<T, G> stratified_proportional_sample(
    const SortedDataSpec<T, G> &data,
    const int                   size) {
    invariant(size > 0, "Sample size must be greater than 0.");
    invariant(size <= data.y.rows(), "Sample size cannot be larger than the number of rows in the data.");

    std::vector<int> sample_indices = models::stats::stratified_proportional_sample(data.x, data.y, data.classes, size);

    return BootstrapDataSpec<T, G>(data.x, data.y, data.classes, sample_indices);
  }
}
