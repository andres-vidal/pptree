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

  template <typename Derived>
  auto mean(const math::DMatrixBase<Derived> &data) {
    return data.colwise().mean().transpose();
  }

  template <typename Derived>
  auto center(const math::DMatrixBase<Derived> &data) {
    return data.rowwise() - mean(data).transpose();
  }

  template <typename T>
  Data<T> covariance(const Data<T> &data) {
    Data<T> centered = center(data);

    return (centered.transpose() * centered) / (data.rows() - 1);
  }

  template <typename T>
  DataColumn<T> sd(const Data<T> &data) {
    return covariance(data).diagonal().array().sqrt();
  }

  template <typename T>
  Data<T> descale(const Data<T> &data) {
    DataColumn<T> scaling_factor = sd(data);

    for (int i = 0; i < scaling_factor.rows(); i++) {
      if (scaling_factor(i) == 0) {
        scaling_factor(i) = 1;
      }
    }

    return data.array().rowwise() / scaling_factor.transpose().array();
  }

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
}
