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

  template<typename T>
  auto outer_product(
    const DMatrixBase<T> &a,
    const DMatrixBase<T> &b
    ) {
    return a * b.transpose();
  }

  template<typename T>
  auto outer_square(
    const DMatrixBase<T> &a
    ) {
    return outer_product(a, a);
  }

  template<typename T>
  double inner_product(
    const DVector<T> &a,
    const DVector<T> &b
    ) {
    return a.transpose()  * b;
  }

  template<typename T>
  double inner_square(
    const DVector<T> &a) {
    return inner_product(a, a);
  }

  template<typename T>
  bool collinear(
    const DVector<T> &a,
    const DVector<T> &b) {
    return is_module_approx(inner_product(a, b) / (a.norm() * b.norm()), 1.0);
  }

  template<typename T>
  DVector<T> abs(
    const DVector<T> &v) {
    return v.array().abs();
  };
}
