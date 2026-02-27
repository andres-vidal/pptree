#pragma once

#include "Types.hpp"
#include "Invariant.hpp"

#include <set>
#include <stdexcept>
#include <pcg_random.hpp>

namespace models::stats {
  using RNG = pcg32;

  types::FeatureMatrix standardize(const types::FeatureMatrix& data);

  void sort(types::FeatureMatrix& x, types::ResponseVector& y);

  std::set<types::Response> unique(const types::ResponseVector& column);

  float accuracy(const types::ResponseVector& predictions, const types::ResponseVector& actual);

  double error_rate(const types::ResponseVector& predictions, const types::ResponseVector& actual);

  double sd(const types::FeatureVector& data);
}
