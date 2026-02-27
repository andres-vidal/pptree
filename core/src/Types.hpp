#pragma once

#include <Eigen/Dense>

namespace models::types {
  #ifdef PPTREE_DOUBLE_PRECISION
  using Feature = double;
  #else
  using Feature = float;
  #endif

  using Response = int;

  using FeatureMatrix = Eigen::Matrix<Feature, Eigen::Dynamic, Eigen::Dynamic>;
  using FeatureVector = Eigen::Matrix<Feature, Eigen::Dynamic, 1>;

  using ResponseVector = Eigen::Matrix<Response, Eigen::Dynamic, 1>;

  template<typename T>
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  template<typename T>
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
}
