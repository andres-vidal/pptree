#pragma once

#include "Random.hpp"
#include "Uniform.hpp"

namespace models::stats {
  class Normal {
    private:
      float mean;
      float std_dev;
      mutable Uniform unif{ 0, 1 };

    public:
      Normal(float mean, float std_dev) : mean(mean), std_dev(std_dev) {
      }

      float operator()() const {
        float u1 = unif();
        float u2 = unif();

        float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
        return mean + std_dev * z;
      }

      std::vector<float> operator()(int count) const {
        std::vector<float> result(count);
        for (int i = 0; i < count; i++) {
          result[i] = operator()();
        }

        return result;
      }
  };
}
