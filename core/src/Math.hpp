#pragma once

#include <cmath>

inline bool is_approx(long double a, long double b) {
  return fabs(a - b) < 0.00001;
}

inline bool is_module_approx(long double a, long double b) {
  return is_approx(fabs(a), fabs(b));
}
