#pragma once
#include <algorithm>
#include "stats.hpp"

namespace dr::strategy {
  template<typename T>
  using DRStrategy = std::function<stats::Data<T>(const stats::Data<T>&)>;

  template<typename T>
  DRStrategy<T> all();

  template<typename T>
  DRStrategy<T> uniform(int n_vars, std::mt19937 &gen);
}
