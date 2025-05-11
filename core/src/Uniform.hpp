#pragma once

#include <random>
#include "Random.hpp"

#include "Invariant.hpp"

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
        invariant(min >= 0, "Uniform: min must be greater than or equal to 0");
        invariant(min <= max, "Uniform: min must be less than or equal to max");
        invariant(max < std::numeric_limits<int>::max(), "Uniform: max must be less than the maximum value of an int");
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

      /**
       * @brief Generates a vector of distinct random integers from the uniform distribution
       *
       * This method implements the Fisher-Yates (Knuth) shuffle algorithm to generate
       * a specified number of unique random integers from the defined range [min, max].
       * The resulting integers are uniformly distributed and guaranteed to be unique.
       *
       * @param count The number of distinct integers to generate
       * @return A vector containing count distinct integers from the range [min, max]
       * @throws std::invalid_argument if count exceeds the number of possible unique values
       *         in the range [min, max]
       */
      std::vector<int> distinct(int count) {
        int range_size = max - min + 1;

        invariant(count <= range_size, "Uniform::distinct: count must be less than or equal to the number of unique values in the range");
        invariant(count >= 0, "Uniform::distinct: count must be greater than or equal to 0");

        std::vector<int> values(range_size);
        std::iota(values.begin(), values.end(), min);

        for (int i = range_size - 1; i >= 0; i--) {
          int j = gen_lemire(i + 1);
          std::swap(values[i], values[j]);
        }

        return std::vector<int>(values.begin(), values.begin() + count);
      }
  };
}
