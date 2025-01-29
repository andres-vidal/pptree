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
  auto inner_product(
    const DMatrixBase<Derived> &a,
    const DMatrixBase<Derived> &b,
    const DMatrixBase<Derived> &weights
    ) {
    return (a.transpose() * weights * b);
  }

  template<typename Derived>
  auto inner_product(
    const DMatrixBase<Derived> &a,
    const DMatrixBase<Derived> &b
    ) {
    return (a.transpose() * b);
  }

  template<typename Derived>
  auto inner_square(
    const DMatrixBase<Derived> &m,
    const DMatrixBase<Derived> &weights
    ) {
    return inner_product(m, m, weights);
  }

  template<typename Derived>
  auto inner_square(
    const DMatrixBase<Derived> &m
    ) {
    return inner_product(m, m);
  }

  template<typename Derived>
  double determinant(
    const DMatrixBase<Derived> &m
    ) {
    return m.determinant();
  }

  template<typename T>
  bool collinear(
    const DMatrix<T> &a,
    const DMatrix<T> &b) {
    for (int i = 0; i < a.cols(); i++) {
      DVector<T> a_col = a.col(i);
      DVector<T> b_col = b.col(i);

      if (!collinear(a_col, b_col)) {
        return false;
      }
    }

    return true;
  }
}
