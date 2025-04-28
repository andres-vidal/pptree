#pragma once

#include "Math.hpp"

#include <Eigen/Dense>
namespace models::math {
  template<typename T>
  using DMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  template<typename Derived>
  using DMatrixBase = Eigen::MatrixBase<Derived>;

  template<typename T>
  using DVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  template<typename Derived>
  DMatrix<float> outer_product(
    const DMatrixBase<Derived> &a,
    const DMatrixBase<Derived> &b
    ) {
    return a * b.transpose();
  }

  template<typename Derived>
  DMatrix<float> outer_square(
    const DMatrixBase<Derived> &a
    ) {
    return outer_product(a, a);
  }

  template<typename T>
  bool collinear(
    const DVector<T> &a,
    const DVector<T> &b) {
    return is_module_approx(a.dot(b) / (a.norm() * b.norm()), 1.0);
  }

  template<typename T>
  DVector<T> abs(
    const DVector<T> &v) {
    return v.array().abs();
  };
}
