/**
 * @file Train.hpp
 * @brief Model training utilities and train subcommand handler.
 *
 * Provides shared functions used by both the train and evaluate
 * subcommands (data loading, model training, configuration display)
 * plus the run_train() entry point.
 */
#pragma once

#include "cli/CLIOptions.hpp"
#include "stats/DataPacket.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "models/Model.hpp"

namespace pptree::cli {
  /** @brief Result of a train operation containing the model and training duration. */
  struct TrainResult {
    pptree::Model::Ptr model;
    long long duration;
  };

  /**
   * @brief Load or simulate data based on CLI options.
   *
   * If data_path is set, reads a CSV file; otherwise generates simulated data.
   * Ensures the response vector is contiguous (sorted by class).
   */
  pptree::stats::DataPacket read_data(
    const CLIOptions&   params,
    pptree::stats::RNG& rng);

  /**
   * @brief Train a single model (Forest or Tree) on the given dataset.
   */
  TrainResult train_model(
    const pptree::types::FeatureMatrix&  x,
    const pptree::types::ResponseVector& y,
    const CLIOptions&                    params,
    pptree::stats::RNG&                  rng);

  /**
   * @brief Print the training configuration summary.
   */
  void announce_configuration(
    const CLIOptions& params,
    int               n_train = 0,
    int               n_test  = 0);

  /**
   * @brief Run the train subcommand.
   * @return Exit code (0 on success).
   */
  int run_train(CLIOptions& params);
}
