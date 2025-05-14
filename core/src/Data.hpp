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
