#include "Random.hpp"

namespace models::stats::Random {
  pcg32 rng{};

  uint32_t min() {
    return 0;
  }

  void seed(const int value) {
    #ifdef _OPENMP
    #pragma omp parallel
    {
      rng.seed(static_cast<uint64_t>(value), static_cast<uint64_t>(omp_get_thread_num()));
    }
    #else
    rng.seed(value, 0u);
    #endif
  }

  uint32_t gen() {
    return rng();
  }
}
