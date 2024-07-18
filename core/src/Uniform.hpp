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

      std::set<int> distinct(int count) {
        std::set<int> result;

        while (result.size() < count)
          result.insert(operator()());

        return result;
      }
  };
}
