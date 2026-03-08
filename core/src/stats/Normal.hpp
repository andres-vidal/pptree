#pragma once

#include "stats/Stats.hpp"
#include "stats/Uniform.hpp"

#include <optional>


namespace pptree::stats {
  /**
   * @brief Normal (Gaussian) random number generator.
   *
   * Generates samples from N(mean, std_dev²) using the Box-Muller
   * transform.  Each call to the RNG produces a pair of independent
   * standard normal variates; the second is cached and returned on
   * the next invocation, so roughly half of the calls avoid the
   * expensive trigonometric computation.
   *
   * Uniform inputs are generated internally at 53-bit precision
   * (see gen_unif01()), matching the significand width of IEEE 754
   * double-precision floats.
   */
  class Normal {
    private:
      float mean;
      float std_dev;

      std::optional<float> cached_z;
      /**
       * @brief Generate a uniform random number in (0, 1).
       *
       * Combines two 32-bit pcg32 outputs into a 53-bit integer and
       * normalizes to the unit interval.  Rejects the boundary values
       * 0 and 1 so that log(U) and log(1−U) are always finite.
       *
       * @param rng  Random number generator (pcg32).
       * @return     Uniform variate in (0, 1).
       */
      double gen_unif01(RNG &rng) {
        static constexpr uint64_t PRECISION = 53;
        static constexpr uint64_t MAX_BITS  = (1ULL << PRECISION) - 1;

        uint64_t bits = 0;

        while (bits == 0 || bits == MAX_BITS) {
          uint64_t x = rng();
          uint64_t y = rng();
          bits = ((x << 21) | (y >> 11)) & MAX_BITS;
        }

        return static_cast<double>(bits) / static_cast<double>(MAX_BITS + 1);
      }

      /**
       * @brief Transform a standard normal variate to N(mean, std_dev²).
       *
       * @param z  Standard normal variate.
       * @return   mean + std_dev * z.
       */
      float denormalize(float z) {
        return mean + std_dev * z;
      }

    public:
      /**
       * @brief Construct a Normal generator.
       *
       * @param mean     Distribution mean.
       * @param std_dev  Distribution standard deviation.
       */
      Normal(float mean, float std_dev) : mean(mean), std_dev(std_dev) {
      }

      /**
       * @brief Generate a single normal variate via the Box-Muller transform.
       *
       * Draws two independent uniform variates U1, U2 ∈ (0, 1) and
       * computes a pair of standard normal variates:
       *
       *   Z1 = √(−2 ln U1) · cos(2π U2)
       *   Z2 = √(−2 ln U1) · sin(2π U2)
       *
       * Returns Z1 (after denormalization) and caches Z2 for the next
       * call.  When a cached value is available it is returned directly
       * without generating new uniform inputs.
       *
       * @param rng  Random number generator (pcg32).
       * @return     A sample from N(mean, std_dev²).
       */
      float operator()(RNG &rng) {
        double u1 = gen_unif01(rng);
        double u2 = gen_unif01(rng);

        double z;

        if (!cached_z.has_value()) {
          double r     = std::sqrt(-2.0 * std::log(u1));
          double theta = 2.0 * M_PI * u2;

          z = r * std::cos(theta);

          double z2 = r * std::sin(theta);

          cached_z = static_cast<float>(z2);
        } else {
          z = cached_z.value();
        }

        return denormalize(z);
      }

      /**
       * @brief Generate multiple normal variates.
       *
       * @param count  Number of samples to generate.
       * @param rng    Random number generator (pcg32).
       * @return       Vector of @p count samples from N(mean, std_dev²).
       */
      std::vector<float> operator()(int count, RNG &rng) {
        std::vector<float> result(count);

        for (int i = 0; i < count; i++) {
          result[i] = operator()(rng);
        }

        return result;
      }
  };
}
