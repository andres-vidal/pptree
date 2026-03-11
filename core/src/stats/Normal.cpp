#include "stats/Normal.hpp"

#include <cmath>

namespace pptree::stats {
  double Normal::gen_unif01(RNG &rng) {
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

  float Normal::denormalize(float z) {
    return mean + std_dev * z;
  }

  Normal::Normal(float mean, float std_dev) : mean(mean), std_dev(std_dev) {
  }

  float Normal::operator()(RNG &rng) {
    if (cached_z.has_value()) {
      float z = cached_z.value();
      cached_z.reset();
      return denormalize(z);
    }

    double u1 = gen_unif01(rng);
    double u2 = gen_unif01(rng);

    double r     = std::sqrt(-2.0 * std::log(u1));
    double theta = 2.0 * M_PI * u2;

    double z1 = r * std::cos(theta);
    double z2 = r * std::sin(theta);

    cached_z = static_cast<float>(z2);

    return denormalize(z1);
  }

  std::vector<float> Normal::operator()(int count, RNG &rng) {
    std::vector<float> result(count);

    for (int i = 0; i < count; i++) {
      result[i] = operator()(rng);
    }

    return result;
  }
}
