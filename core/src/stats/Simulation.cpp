#include "stats/Simulation.hpp"
#include "stats/Normal.hpp"
#include "stats/Uniform.hpp"
#include "stats/GroupPartition.hpp"

namespace pptree::stats {
  DataPacket simulate(
    const int               n,
    const int               p,
    const int               G,
    RNG&                    rng,
    const SimulationParams& params) {
    types::FeatureMatrix x(n, p);
    types::ResponseVector y(n);

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

  Split split(const DataPacket& data, float train_ratio, RNG& rng) {
    const int n          = data.x.rows();
    const int train_size = static_cast<int>(n * train_ratio);

    GroupPartition spec(data.y);

    std::vector<int> train_indices;
    std::vector<int> test_indices;

    train_indices.reserve(train_size);
    test_indices.reserve(n - train_size);

    for (const auto& group : data.classes) {
      int group_start      = spec.group_start(group);
      int group_size       = spec.group_size(group);
      int group_end        = group_start + group_size - 1;
      int group_train_size = static_cast<int>(group_size * train_ratio);

      Uniform unif(group_start, group_end);
      std::vector<int> group_indices = unif.distinct(group_size, rng);

      train_indices.insert(train_indices.end(), group_indices.begin(), group_indices.begin() + group_train_size);
      test_indices.insert(test_indices.end(), group_indices.begin() + group_train_size, group_indices.end());
    }

    return {
      .tr = train_indices,
      .te = test_indices
    };
  }
}
