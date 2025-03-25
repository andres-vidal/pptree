#pragma once

#include <random>
#include "Random.hpp"
namespace models::stats {
  class Uniform {
    private:
      int min;
      int max;

    public:
      Uniform(int min, int max) : min(min), max(max) {
      }

      int operator()() const {
        uint64_t range = static_cast<uint64_t>(max) - min + 1;
        uint64_t random_number = Random::gen() - Random::min();
        return min + static_cast<int>(random_number % range);
      }

      std::vector<int> operator()(int count) const {
        std::vector<int> result(count);

        for (int i = 0; i < count; i++) {
          result[i] = operator()();
        }

        return result;
      }

      std::vector<int> distinct(int count) {
        int range_size = max - min + 1;

        if (count > range_size) {
          throw std::invalid_argument("Count exceeds the number of unique values in the range");
        }

        std::vector<int> values(range_size);
        std::iota(values.begin(), values.end(), min);

        for (int i = range_size - 1; i > 0; i--) {
          int j = operator()() % (i + 1);
          std::swap(values[i], values[j]);
        }

        return std::vector<int>(values.begin(), values.begin() + count);
      }
  };
}
