#pragma once

#include "DataSpec.hpp"
#include "Uniform.hpp"
namespace models::stats {
  template<typename T, typename G>
  struct BootstrapDataSpec : DataSpec<T, G> {
    const std::vector<int> sample_indices;
    const std::set<int> oob_indices;

    BootstrapDataSpec(
      const Data<T> &         x,
      const DataColumn<G> &   y,
      const std::set<G> &     classes,
      const std::vector<int> &sample_indices)
      : DataSpec<T, G>(x, y, classes),
      sample_indices(sample_indices),
      oob_indices(init_oob_indices(x, sample_indices)) {
    }

    BootstrapDataSpec(
      const Data<T> &         x,
      const DataColumn<G> &   y,
      const std::vector<int> &sample_indices)
      : DataSpec<T, G>(x, y),
      sample_indices(sample_indices),
      oob_indices(init_oob_indices(x, sample_indices)) {
    }

    DataSpec<T, G> get_sample() const {
      return DataSpec<T, G>(
        select_rows(this->x, this->sample_indices),
        select_rows(this->y, this->sample_indices));
    }

    DataSpec<T, G> get_oob() const {
      return DataSpec<T, G>(
        select_rows(this->x, this->oob_indices),
        select_rows(this->y, this->oob_indices));
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
    const DataSpec<T, G> &data,
    const int             size) {
    assert(size > 0 && "Sample size must be greater than 0.");
    assert(size <= data.y.rows() && "Sample size cannot be larger than the number of rows in the data.");

    const int data_size = data.y.rows();

    std::vector<int> sample_indices;

    for (const G& group : data.classes) {
      const std::vector<int> group_indices = select_group(data.y, group);

      const int group_size = group_indices.size();
      const int group_sample_size = std::round(group_size / (double)data_size * size);

      for (int i = 0; i < group_sample_size; i++) {
        const Uniform unif(0, group_indices.size() - 1);
        const int sampled_index = group_indices[unif()];
        sample_indices.push_back(sampled_index);
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
