#pragma once

#include "Error.hpp"

#include <random>

#ifdef _OPENMP
#include "omp.h"
#endif

namespace models::stats {
  class Random {
    public:
      static std::vector<std::mt19937> rngs;

      static uint_fast32_t min() {
        return get_current_rng().min();
      }

      static void seed(const uint_fast32_t value) {
        #ifdef _OPENMP

        if (omp_get_thread_num() > 0) {
          throw std::runtime_error("Random::seed() must be called from the master thread.");
        }

        for (int i = 0; i < rngs.size(); i++) {
          rngs[i].seed(value + i);
        }

        #else

        get_current_rng().seed(value);

        #endif
      }

      static uint_fast32_t gen() {
        std::mt19937& rng = get_current_rng();
        return rng();
      }

      static std::mt19937& get_current_rng() {
        std::mt19937& rng = rngs[0];

        #ifdef _OPENMP
        rng = rngs[omp_get_thread_num()];
        #endif

        return rng;
      }
  };
}
