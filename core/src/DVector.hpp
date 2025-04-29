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
  bool collinear(const DVector<T> &a, const DVector<T> &b) {
    return is_module_approx(a.dot(b) / (a.norm() * b.norm()), 1.0);
  }
}
