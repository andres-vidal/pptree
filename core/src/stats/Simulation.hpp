#pragma once

#include "stats/Stats.hpp"
#include "stats/DataPacket.hpp"
#include "utils/Types.hpp"

#include <vector>

namespace ppforest2::stats {
  /**
   * @brief Parameters for generating simulated classification data.
   *
   * Controls the distribution of generated features across groups.
   * Each group is drawn from a normal distribution with a shifted mean
   * relative to the previous group.
   */
  struct SimulationParams {
    types::Feature mean            = 100.0f; ///< Base mean for the first group.
    types::Feature mean_separation = 50.0f;  ///< Mean shift between successive groups.
    types::Feature sd              = 10.0f;  ///< Standard deviation within each group.
  };

  /**
   * @brief Generate a simulated dataset with G groups, n rows, and p features.
   *
   * Each group is drawn from a normal distribution with a shifted mean.
   * The resulting data is sorted by group label.
   *
   * @param n      Number of rows (observations).
   * @param p      Number of feature columns.
   * @param G      Number of groups (must be > 1).
   * @param rng    Random number generator.
   * @param params Simulation parameters (mean, separation, sd).
   * @return A DataPacket with the simulated feature matrix and response vector.
   */
  DataPacket simulate(int n, int p, int G, RNG& rng, SimulationParams const& params = SimulationParams{});

  /**
   * @brief Indices for a train/test split.
   */
  struct Split {
    std::vector<int> tr; ///< Training set indices.
    std::vector<int> te; ///< Test set indices.
  };

  /**
   * @brief Perform a stratified random train/test split on a DataPacket.
   *
   * Samples indices within each group proportional to train_ratio so that
   * group balance is preserved in both train and test sets.
   *
   * @param data        The full dataset.
   * @param train_ratio Proportion of data to use for training (0, 1).
   * @param rng         Random number generator.
   * @return A Split containing train and test index vectors.
   */
  Split split(DataPacket const& data, float train_ratio, RNG& rng);
}
