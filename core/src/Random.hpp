#pragma once

#include <random>

namespace models::stats {
  class Random {
    public:
      static std::mt19937 rng;


      static uint_fast32_t gen() {
        uint_fast32_t value;

        #pragma omp critical
        { value = rng(); }
        return value;
      }
  };
}
