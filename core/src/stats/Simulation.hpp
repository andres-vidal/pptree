#pragma once

#include "stats/Stats.hpp"
#include "stats/DataPacket.hpp"
#include "utils/Types.hpp"

#include <vector>

namespace pptree::stats {
  /**
   * @brief Parameters for generating simulated classification data.
   *
   * Controls the distribution of generated features across classes.
   * Each class is drawn from a normal distribution with a shifted mean
   * relative to the previous class.
   */
  struct SimulationParams {
    float mean            = 100.0f;   ///< Base mean for the first class.
    float mean_separation = 50.0f;    ///< Mean shift between successive classes.
    float sd              = 10.0f;    ///< Standard deviation within each class.
  };

  /**
   * @brief Generate a simulated dataset with G classes, n rows, and p features.
   *
   * Each class is drawn from a normal distribution with a shifted mean.
   * The resulting data is sorted by class label.
   *
   * @param n      Number of rows (observations).
   * @param p      Number of feature columns.
   * @param G      Number of classes (must be > 1).
   * @param rng    Random number generator.
   * @param params Simulation parameters (mean, separation, sd).
   * @return A DataPacket with the simulated feature matrix and response vector.
   */
  DataPacket simulate(
    int                     n,
    int                     p,
    int                     G,
    RNG&                    rng,
    const SimulationParams& params = SimulationParams{});

  /**
   * @brief Indices for a train/test split.
   */
  struct Split {
    std::vector<int> tr;   ///< Training set indices.
    std::vector<int> te;   ///< Test set indices.
  };

  /**
   * @brief Perform a stratified random train/test split on a DataPacket.
   *
   * Samples indices within each class proportional to train_ratio so that
   * class balance is preserved in both train and test sets.
   *
   * @param data        The full dataset.
   * @param train_ratio Proportion of data to use for training (0, 1).
   * @param rng         Random number generator.
   * @return A Split containing train and test index vectors.
   */
  Split split(const DataPacket& data, float train_ratio, RNG& rng);
}
