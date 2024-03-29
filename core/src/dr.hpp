#pragma once
#include <algorithm>
#include "stats.hpp"

namespace dr::strategy {
  template<typename T>
  using DRStrategy = std::function<stats::Data<T>(const stats::Data<T>&)>;

  template<typename T>
  DRStrategy<T> select_all_variables();

  template<typename T>
  DRStrategy<T> select_variables_uniformly(int n_vars, std::mt19937 &gen);
}
