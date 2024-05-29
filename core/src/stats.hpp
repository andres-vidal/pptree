#pragma once

#include "linalg.hpp"
#include <set>
#include <map>
#include <random>
#include <algorithm>
#include <vector>


namespace stats {
  class Uniform {
    private:
      int min;
      int max;

    public:
      Uniform(int min, int max) : min(min), max(max) {
      }

      int operator()(std::mt19937 &rng) const {
        uint64_t range = static_cast<uint64_t>(max) - min + 1;
        uint64_t random_number = rng() - rng.min();
        return min + static_cast<int>(random_number % range);
      }

      std::vector<int> operator()(std::mt19937 &rng, int count) const {
        std::vector<int> result(count);

        for (int i = 0; i < count; i++) {
          result[i] = operator()(rng);
        }

        return result;
      }
  };

  template<typename T>
  using Data = linalg::DMatrix<T>;

  template<typename T>
  using DataColumn = linalg::DVector<T>;

  template<typename N>
  std::set<N> unique(const DataColumn<N> &column);

  template<typename T>
  Data<T> select_rows(
    const Data<T> &         data,
    const std::vector<int> &indices);

  template<typename T>
  DataColumn<T> select_rows(
    const DataColumn<T> &   data,
    const std::vector<int> &indices);

  template<typename T>
  Data<T> select_rows(
    const Data<T> &      data,
    const std::set<int> &indices);

  template<typename T>
  DataColumn<T> select_rows(
    const DataColumn<T> & data,
    const std::set<int> & indices);

  template<typename T, typename G>
  struct DataSpec {
    const Data<T>  x;
    const DataColumn<G>  y;
    const std::set<G>  classes;

    DataSpec(
      const Data<T> &       x,
      const DataColumn<G> & y,
      const std::set<G> &   classes)
      : x(x),
      y(y),
      classes(classes) {
    }

    DataSpec(
      const Data<T> &       x,
      const DataColumn<G> & y)
      : x(x),
      y(y),
      classes(unique(y)) {
    }

    virtual std::tuple<Data<T>, DataColumn<G>, std::set<G> > unwrap() const {
      return std::make_tuple(x, y, classes);
    }
  };

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

  template<typename G>
  std::vector<G> select_group(
    const DataColumn<G> &groups,
    const G &            group);

  template<typename T, typename G>
  Data<T> select_group(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const G &            group);

  template<typename T, typename G>
  DataColumn<T> select_group(
    const DataColumn<T> &data,
    const DataColumn<G> &groups,
    const G &            group);

  template<typename T, typename G>
  Data<T> select_groups(
    const Data<T> &      data,
    const DataColumn<G> &data_groups,
    const std::set<G> &  groups);

  template<typename T, typename G>
  Data<T> remove_group(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const G &            group);

  template<typename T, typename G>
  std::tuple<DataColumn<G>, std::set<int>, std::map<int, std::set<G> > >binary_regroup(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups);

  template<typename T, typename G>
  Data<T> between_groups_sum_of_squares(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups);


  template<typename T, typename G>
  Data<T> within_groups_sum_of_squares(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups);

  template<typename T, typename G>
  BootstrapDataSpec<T, G> stratified_proportional_sample(
    const DataSpec<T, G> &data,
    const int             size,
    std::mt19937 &        rng);

  template<typename T>
  std::tuple<std::vector<int>, std::vector<int> > mask_null_columns(
    const Data<T> &data);

  template<typename T>
  DataColumn<T> expand(
    const DataColumn<T> &   data,
    const std::vector<int> &mask);

  template<typename T>
  DataColumn<T> mean(
    const Data<T> &data);

  template<typename T>
  T mean(
    const DataColumn<T> &data);

  template<typename T>
  Data<T> covariance(
    const Data<T> &data);

  template<typename T>
  DataColumn<T> sd(
    const Data<T> &data);

  template<typename T>
  T sd(
    const DataColumn<T> &data);

  template<typename T>
  Data<T> center(
    const Data<T> &data);

  template<typename T>
  DataColumn<T> center(
    const DataColumn<T> &data);

  template<typename T, typename G>
  DataSpec<T, G> center(
    const DataSpec<T, G> &data);

  template<typename T, typename G>
  BootstrapDataSpec<T, G> center(
    const BootstrapDataSpec<T, G> &data);

  template<typename T>
  Data<T> descale(
    const Data<T> &data);

  template<typename T>
  DataColumn<T> descale(
    const DataColumn<T> &data);

  template<typename T, typename G>
  DataSpec<T, G> descale(
    const DataSpec<T, G> &data);

  template<typename T, typename G>
  BootstrapDataSpec<T, G> descale(
    const BootstrapDataSpec<T, G> &data);

  template<typename T>
  Data<T> shuffle_column(
    const Data<T> &data,
    const int      column,
    std::mt19937 & rng);
};
