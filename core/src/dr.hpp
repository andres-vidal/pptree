#pragma once
#include <algorithm>

#include "Data.hpp"
#include "Uniform.hpp"

namespace dr::strategy {
  template<typename T>
  using DRStrategy = std::function<Data<T>(const Data<T>, std::mt19937 &rng)>;

  template<typename T>
  DRStrategy<T> all();

  template<typename T>
  DRStrategy<T> uniform(int n_vars);
}
