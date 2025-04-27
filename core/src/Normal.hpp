#pragma once

#include "Random.hpp"
#include "Uniform.hpp"

#include <optional>


namespace models::stats {
  class Normal {
    private:
      float mean;
      float std_dev;

      std::optional<float> cached_z;

      double gen_unif01() {
        static constexpr uint64_t PRECISION = 53;
        static constexpr uint64_t MAX_BITS = (1ULL << PRECISION) - 1;

        uint64_t bits = 0;

        while (bits == 0 || bits == MAX_BITS) {
          uint64_t x = Random::gen();
          uint64_t y = Random::gen();
          bits = ((x << 21) | (y >> 11)) & MAX_BITS;
        }

        return static_cast<double>(bits) / static_cast<double>(MAX_BITS + 1);
      }

      float denormalize(float z) {
        return mean + std_dev * z;
      }

    public:
      Normal(float mean, float std_dev) : mean(mean), std_dev(std_dev) {
      }

      /**
       * @brief Generates a normally distributed random number using the Box-Muller transform.
       *
       * This implementation uses the polar form of the Box-Muller transform to convert
       * uniform random variables into normally distributed ones. The algorithm works by
       * generating pairs of independent standard normal random variables (Z1, Z2) using:
       *
       * Z1 = R * cos(θ)
       * Z2 = R * sin(θ)
       *
       * where:
       * - R = sqrt(-2 * ln(U1))  [radius]
       * - θ = 2π * U2            [angle]
       * - U1, U2 are uniform random variables in (0,1)
       *
       * For efficiency, this implementation caches the second generated value (Z2)
       * for the next call.
       *
       * @return A random float from the normal distribution with specified mean and standard deviation
       */
      float operator()() {
        double u1 = gen_unif01();
        double u2 = gen_unif01();

        double z;

        if (!cached_z.has_value()) {
          double r = std::sqrt(-2.0 * std::log(u1));
          double theta = 2.0 * M_PI * u2;

          z = r * std::cos(theta);

          double z2 = r * std::sin(theta);

          cached_z = static_cast<float>(z2);
        } else {
          z = cached_z.value();
        }

        return denormalize(z);
      }

      std::vector<float> operator()(int count) {
        std::vector<float> result(count);

        for (int i = 0; i < count; i++) {
          result[i] = operator()();
        }

        return result;
      }
  };
}
