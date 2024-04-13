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
      uint64_t range;

    public:
      Uniform(int min, int max) : min(min), max(max), range(static_cast<uint64_t>(max) - min + 1) {
      }

      int operator()(std::mt19937 &rng) const {
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

  template<typename T, typename R>
  struct DataSpec {
    const Data<T>  &x;
    const DataColumn<R>  &y;
    const std::set<R>  classes;

    DataSpec(
      const Data<T> &       x,
      const DataColumn<R> & y,
      const std::set<R>     classes)
      : x(x),
      y(y),
      classes(classes) {
    }

    DataSpec(
      const Data<T> &       x,
      const DataColumn<R> & y)
      : x(x),
      y(y),
      classes(unique(y)) {
    }
  };

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

  template<typename T>
  Data<T> sample(
    const Data<T> & data,
    int             size,
    std::mt19937 &  rng);

  template<typename T, typename G>
  std::tuple<Data<T>, DataColumn<G> > stratified_sample(
    const Data<T> &        data,
    const DataColumn<G> &  groups,
    const std::map<G, int> sizes,
    std::mt19937 &         rng);

  template<typename T, typename G>
  std::tuple<Data<T>, DataColumn<G> > stratified_proportional_sample(
    const Data<T> &       data,
    const DataColumn<G> & groups,
    const std::set<G> &   unique_groups,
    const int             size,
    std::mt19937 &        rng);

  template<typename T>
  std::tuple<std::vector<int>, std::vector<int> > mask_null_columns(
    const Data<T> &data);

  template<typename T>
  DataColumn<T> expand(
    const DataColumn<T> &   data,
    const std::vector<int> &mask);
};
