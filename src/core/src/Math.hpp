#pragma once

#include <cmath>

#include "Macros.hpp"

namespace models::math {
  inline bool is_approx(float a, float b, float threshold) {
    return fabs(a - b) < threshold;
  }

  inline bool is_approx(float a, float b) {
    return is_approx(a, b, APPROX_THRESHOLD);
  }

  inline bool is_module_approx(float a, float b) {
    return is_approx(fabs(a), fabs(b));
  }
}
