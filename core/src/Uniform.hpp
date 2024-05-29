#pragma once

#include <random>

class Uniform {
  private:
    int min;
    int max;

  public:
    Uniform(int min, int max) : min(min), max(max) {
    }

    int operator()(std::mt19937 &rng) const {
      uint64_t range = static_cast<uint64_t>(max) - min + 1;
      uint64_t random_number = rng() - rng.min();
      return min + static_cast<int>(random_number % range);
    }

    std::vector<int> operator()(std::mt19937 &rng, int count) const {
      std::vector<int> result(count);

      for (int i = 0; i < count; i++) {
        result[i] = operator()(rng);
      }

      return result;
    }
};
