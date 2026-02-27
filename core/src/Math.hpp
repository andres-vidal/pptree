#pragma once

#include <cmath>

#include "Types.hpp"
#include "Macros.hpp"

#define APPROX_THRESHOLD 0.01

namespace models::math {
  template<typename A, typename B, typename T>
  inline bool is_approx(A a, B b, T threshold) {
    return fabs(a - b) < threshold;
  }

  template<typename A, typename B>
  inline bool is_approx(A a, B b) {
    return is_approx(a, b, APPROX_THRESHOLD);
  }

  template<typename A, typename B>
  inline bool is_module_approx(A a, B b) {
    return is_approx(fabs(a), fabs(b));
  }

  template<typename T>
  bool collinear(const types::Vector<T> &a, const types::Vector<T> &b) {
    return is_module_approx(a.dot(b) / (a.norm() * b.norm()), 1.0);
  }
}
