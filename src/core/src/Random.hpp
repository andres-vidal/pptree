#pragma once

#include <random>

namespace models::stats {
  class Random {
    public:
      static std::mt19937 rng;
  };
}
