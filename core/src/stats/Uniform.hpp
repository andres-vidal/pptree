#pragma once

#include <vector>
#include "stats/Stats.hpp"

namespace ppforest2::stats {
  /**
   * @brief Discrete uniform random integer generator over [min, max].
   *
   * All random integers are produced via Lemire's nearly-divisionless
   * method, which replaces the classical modulo reduction with a
   * multiply-and-shift, falling back to rejection sampling only when
   * needed to eliminate bias.  This is both faster and provably
   * unbiased.
   *
   * The distinct() method builds on this by performing a Fisher-Yates
   * (Knuth) shuffle to sample without replacement — the foundation
   * of bootstrap sampling and random variable selection in forests.
   *
   * @note This class is used instead of std::uniform_int_distribution
   *       to guarantee identical output across compilers and platforms.
   *
   * @see https://arxiv.org/abs/1805.10941
   */
  class Uniform {
    private:
      int min;
      int max;

      /**
       * @brief Generate an unbiased random integer in [0, s) via Lemire's method.
       *
       * Computes a 64-bit product x·s where x is a 32-bit pcg32 output.
       * The upper 32 bits of the product give the result.  When the lower
       * 32 bits (the remainder analogue) fall below s, rejection sampling
       * kicks in to correct the bias — but this happens with probability
       * < s / 2³², so for small s it is essentially free.
       *
       * @param s    Exclusive upper bound (must be > 0).
       * @param rng  Random number generator (pcg32).
       * @return     Uniformly distributed integer in [0, s−1].
       *
       * @see Lemire, "Fast Random Integer Generation in an Interval"
       *      (ACM TOMS, 2019). https://arxiv.org/abs/1805.10941
       */
      uint32_t gen_lemire(uint32_t s, RNG &rng) const;

    public:
      /**
       * @brief Construct a uniform integer generator over [min, max].
       *
       * @param min  Inclusive lower bound (must be ≥ 0).
       * @param max  Inclusive upper bound (must be ≥ min).
       */
      Uniform(int min, int max);

      /**
       * @brief Generate a single uniform random integer in [min, max].
       *
       * @param rng  Random number generator (pcg32).
       * @return     Uniformly distributed integer in [min, max].
       */
      int operator()(RNG &rng) const;

      /**
       * @brief Generate multiple uniform random integers (with replacement).
       *
       * @param count  Number of samples to generate.
       * @param rng    Random number generator (pcg32).
       * @return       Vector of @p count i.i.d. integers from [min, max].
       */
      std::vector<int> operator()(int count, RNG &rng) const;

      /**
       * @brief Sample without replacement from [min, max].
       *
       * Implements the Fisher-Yates (Knuth) shuffle: fills a vector
       * with [min, max], shuffles it using gen_lemire() for each swap,
       * and returns the first @p count elements.  The result is a
       * uniformly distributed subset of size @p count with no repeats.
       *
       * This is the method used by DRUniformStrategy and bootstrap
       * sampling, ensuring reproducible variable selection across
       * platforms.
       *
       * @param count  Number of distinct integers to draw (≤ range size).
       * @param rng    Random number generator (pcg32).
       * @return       Vector of @p count distinct integers from [min, max].
       */
      std::vector<int> distinct(int count, RNG &rng);
  };
}
