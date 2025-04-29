#pragma once

#include "Math.hpp"
#include "DMatrix.hpp"

#include <Eigen/Dense>
namespace models::math {
  template<typename T>
  using DVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  template<typename T>
  bool collinear(const DVector<T> &a, const DVector<T> &b) {
    return is_module_approx(a.dot(b) / (a.norm() * b.norm()), 1.0);
  }
}
