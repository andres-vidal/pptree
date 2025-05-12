#pragma once
#include <random>
#include <pcg_random.hpp>

#ifdef _OPENMP
#include "omp.h"
#endif

namespace models::stats::Random {
  extern pcg32 rng;
  #pragma omp threadprivate(rng)

  uint32_t min();

  void seed(const int value);

  uint32_t gen();
}
