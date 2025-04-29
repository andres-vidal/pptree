#pragma once

#include "DMatrix.hpp"
#include "DataColumn.hpp"
#include "Uniform.hpp"

#include <map>
#include <set>
#include <random>

namespace models::stats
{
  template <typename T>
  using Data = math::DMatrix<T>;

  template <typename T>
  using DataView = Eigen::Block<const Data<T> >;

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
}
