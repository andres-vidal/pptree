#include "stats/Simulation.hpp"
#include "stats/Normal.hpp"
#include "stats/Uniform.hpp"
#include "stats/GroupPartition.hpp"

namespace ppforest2::stats {
  DataPacket simulate(int const n, int const p, int const G, RNG& rng, SimulationParams const& params) {
    types::FeatureMatrix x(n, p);
    types::OutcomeVector y(n);

    for (int i = 0; i < n; ++i) {
      float group_mean = params.mean + (i % G) * params.mean_separation;

      Normal norm(group_mean, params.sd);

      for (int j = 0; j < p; ++j) {
        x(i, j) = norm(rng);
      }

      y[i] = i % G;
    }

    sort(x, y);

    return DataPacket(x, y);
  }

  Split split(DataPacket const& data, float train_ratio, RNG& rng) {
    int const n          = data.x.rows();
    int const train_size = static_cast<int>(n * train_ratio);

    GroupPartition spec(data.y);

    std::vector<int> train_indices;
    std::vector<int> test_indices;

    train_indices.reserve(train_size);
    test_indices.reserve(n - train_size);

    for (auto const& group : data.groups) {
      int group_start      = spec.group_start(group);
      int group_size       = spec.group_size(group);
      int group_end        = group_start + group_size - 1;
      int group_train_size = static_cast<int>(group_size * train_ratio);

      Uniform unif(group_start, group_end);
      std::vector<int> group_indices = unif.distinct(group_size, rng);

      train_indices.insert(train_indices.end(), group_indices.begin(), group_indices.begin() + group_train_size);
      test_indices.insert(test_indices.end(), group_indices.begin() + group_train_size, group_indices.end());
    }

    return {.tr = train_indices, .te = test_indices};
  }
}
