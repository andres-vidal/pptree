#pragma once

#include <cmath>

namespace models::math {
  inline bool is_approx(double a, double b) {
    return fabs(a - b) < 0.00001;
  }

  inline bool is_module_approx(double a, double b) {
    return is_approx(fabs(a), fabs(b));
  }

  template<typename T>
  inline T truncate(T value) {
    return fabs(value) < 1e-15 ? 0 : value;
  }
}
