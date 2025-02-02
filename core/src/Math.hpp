#pragma once

#include <cmath>

#include "Macros.hpp"

namespace models::math {
  inline bool is_approx(double a, double b, double threshold) {
    return fabs(a - b) < threshold;
  }

  inline bool is_approx(double a, double b) {
    return is_approx(a, b, APPROX_THRESHOLD);
  }

  inline bool is_module_approx(double a, double b) {
    return is_approx(fabs(a), fabs(b));
  }

  template<typename T>
  inline T truncate(T value) {
    return fabs(value) < 1e-15 ? 0 : value;
  }
}
