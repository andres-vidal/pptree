#pragma once

#include <random>
#include "Random.hpp"

namespace models::stats {
  class Uniform {
    private:
      int min;
      int max;

      /**
       * @brief Generates an unbiased random integer in range [0, s-1] using Lemire's method
       *
       * Implementation of Daniel Lemire's fast and unbiased algorithm for random integers
       * in an interval, as described in "Fast Random Integer Generation in an Interval"
       * (ACM TOMS, 2019).
       *
       * The algorithm uses multiplication and bit shifts instead of division or modulo
       * for better performance. It employs rejection sampling when necessary to ensure
       * unbiased results.
       *
       * @param s The exclusive upper bound for the generated numbers (must be > 0)
       * @return A uniformly distributed random integer in [0, s-1]
       * @see https://arxiv.org/abs/1805.10941
       */
      int gen_lemire(uint32_t s) const {
        uint32_t x = Random::gen();
        uint64_t m = static_cast<uint64_t>(x) * s;
        uint32_t l = static_cast<uint32_t>(m);

        if (l < s) {
          uint32_t t = -s % s;

          while (l < t) {
            x = Random::gen();
            m = static_cast<uint64_t>(x) * s;
            l = static_cast<uint32_t>(m);
          }
        }

        return m >> 32;
      }

    public:
      Uniform(int min, int max) : min(min), max(max) {
      }

      int operator()() const {
        return min + gen_lemire(max + 1 - min);
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
