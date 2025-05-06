#pragma once

#include "DMatrix.hpp"
#include "DataColumn.hpp"

#include "GroupSpec.hpp"
#include "Uniform.hpp"

#include "Invariant.hpp"

#include <map>
#include <set>
#include <random>

namespace models::stats
{
  template <typename T>
  using Data = math::DMatrix<T>;

  template <typename T>
  Data<T> shuffle_column(
    const Data<T> &data,
    const int      column) {
    Data<T> shuffled = data;

    Uniform unif(0, data.rows() - 1);

    std::vector<int> indices(data.rows());
    std::iota(indices.begin(), indices.end(), 0);

    for (int i = data.rows() - 1; i > 0; i--) {
      int j = unif();
      std::swap(indices[i], indices[j]);
    }

    for (int i = 0; i < data.rows(); i++) {
      shuffled(i, column) = data(indices[i], column);
    }

    return shuffled;
  }

  template <typename T>
  Data<T> standardize(const Data<T> &data) {
    Data<T> centered = data.rowwise() - data.colwise().mean();
    DataColumn<T> sd = (centered.array().square().colwise().sum() / (data.rows() - 1)).sqrt();

    return centered.array().rowwise() / sd.transpose().array();
  }

  template <typename T, typename R>
  void sort(Data<T> &x, DataColumn<R> &y) {
    std::vector<int> indices(x.rows());
    std::iota(indices.begin(), indices.end(), 0);

    std::stable_sort(
      indices.begin(),
      indices.end(),
      [&y](int idx1, int idx2) {
        return y(idx1) < y(idx2);
      });

    x = x(indices, Eigen::all).eval();
    y = y(indices, Eigen::all).eval();
  }

  template<typename T, typename G>
  std::vector<int> stratified_proportional_sample(
    const Data<T> &      x,
    const DataColumn<G> &y,
    const std::set<G> &  classes,
    const int            size) {
    invariant(size > 0, "Sample size must be greater than 0.");
    invariant(size <= y.rows(), "Sample size cannot be larger than the number of rows in the data.");

    GroupSpec<G> spec(y);

    const int data_size = y.rows();

    std::vector<int> iob_indices;
    iob_indices.reserve(size);

    for (const G& group : classes) {
      const int group_size        = spec.group_size(group);
      const int group_sample_size = std::max(1, (int)std::round(group_size / (float)data_size * size));
      const Uniform unif(spec.group_start(group), spec.group_end(group));

      for (int i = 0; i < group_sample_size; i++) {
        iob_indices.push_back(unif());
      }
    }

    std::sort(iob_indices.begin(), iob_indices.end());

    return iob_indices;
  }
}
