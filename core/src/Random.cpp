#include "Random.hpp"

namespace models::stats::Random {
  std::mt19937 rng{};

  uint_fast32_t min() {
    return rng.min();
  }

  void seed(const uint_fast32_t value) {
    #ifdef _OPENMP
    rng.seed(value + omp_get_thread_num());
    #else
    rng.seed(value);
    #endif
  }

  uint_fast32_t gen() {
    return rng();
  }
}
