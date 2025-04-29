#pragma once

#include "Logger.hpp"
#include "DVector.hpp"

#include <Eigen/Dense>

namespace models::math {
  template<typename T>
  using DMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  template<typename T>
  using DVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
}
