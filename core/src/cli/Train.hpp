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
#include "io/Output.hpp"
#include "stats/DataPacket.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "models/Model.hpp"

namespace CLI {
  class App;
}

namespace ppforest2::cli {
  /** @brief Register train subcommand options on @p app. */
  void setup_train(CLI::App& app, Params& params);

  /** @brief Add shared model options (size, lambda, threads, seed, vars) to @p sub. */
  void add_model_options(CLI::App* sub, ModelParams& model);

  /** @brief Result of a train operation containing the model and training duration. */
  struct TrainResult {
    ppforest2::Model::Ptr model;
    long long duration;
  };

  /**
   * @brief Load or simulate data based on CLI options.
   *
   * If data_path is set, reads a CSV file; otherwise generates simulated data.
   * Ensures the response vector is contiguous (sorted by group).
   */
  ppforest2::stats::DataPacket read_data(Params const& params, ppforest2::stats::RNG& rng);

  /**
   * @brief Train a single model (Forest or Tree) on the given dataset.
   */
  TrainResult train_model(
      ppforest2::types::FeatureMatrix const& x,
      ppforest2::types::OutcomeVector const& y,
      Params const& params,
      ppforest2::stats::RNG& rng
  );

  /**
   * @brief Run the train subcommand.
   * @return Exit code (0 on success).
   */
  int run_train(Params& params);
}
