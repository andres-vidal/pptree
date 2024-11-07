#pragma once
#include <random>

#ifdef _OPENMP
#include "omp.h"
#endif

namespace models::stats::Random {
  extern std::mt19937 rng;
  #pragma omp threadprivate(rng)

  uint_fast32_t min();

  void seed(const uint_fast32_t value);

  uint_fast32_t gen();
}
