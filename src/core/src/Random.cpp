#include "Random.hpp"

namespace models::stats::Random {
  pcg32 rng{};

  uint_fast32_t min() {
    return 0;
  }

  void seed(const uint_fast32_t value) {
    #ifdef _OPENMP
    #pragma omp parallel
    {
      rng.seed(value, static_cast<uint64_t>(omp_get_thread_num()));
    }
    #else
    rng.seed(value, 0u);
    #endif
  }

  uint_fast32_t gen() {
    return rng();
  }
}
