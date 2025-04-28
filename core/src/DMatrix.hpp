#pragma once

#include "Logger.hpp"
#include "DVector.hpp"

#include <Eigen/Dense>

namespace models::math {
  template<typename T>
  using DMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  template<typename Derived>
  using DMatrixBase = Eigen::MatrixBase<Derived>;

  template<typename T>
  using DVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  template<typename Derived>
  DMatrix<float> inner_product(
    const DMatrixBase<Derived> &a,
    const DMatrixBase<Derived> &b,
    const DMatrixBase<Derived> &weights
    ) {
    return (a.transpose() * weights * b);
  }

  template<typename Derived>
  DMatrix<float> inner_product(
    const DMatrixBase<Derived> &a,
    const DMatrixBase<Derived> &b
    ) {
    return (a.transpose() * b);
  }

  template<typename Derived>
  DMatrix<float> inner_square(
    const DMatrixBase<Derived> &m,
    const DMatrixBase<Derived> &weights
    ) {
    return inner_product(m, m, weights);
  }

  template<typename Derived>
  DMatrix<float> inner_square(
    const DMatrixBase<Derived> &m
    ) {
    return inner_product(m, m);
  }

  template<typename Derived>
  float determinant(
    const DMatrixBase<Derived> &m
    ) {
    float d = m.determinant();

    return fabs(d) < 1e-15 ? 0.0 : d;
  }

  template<typename Derived>
  DMatrix<float> truncate(
    const DMatrixBase<Derived> &m
    ) {
    return (m.array().abs() < 1e-15).select(0, m.array());
  }
}
