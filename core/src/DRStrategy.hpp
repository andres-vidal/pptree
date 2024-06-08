#pragma once
#include <algorithm>

#include "Data.hpp"
#include "Uniform.hpp"

namespace models::dr::strategy {
  template<typename T>
  using DRStrategy = std::function<stats::Data<T>(const stats::Data<T>, std::mt19937 &rng)>;

  template<typename T>
  DRStrategy<T> all();

  template<typename T>
  DRStrategy<T> uniform(int n_vars);
}
