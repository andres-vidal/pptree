/**
 * @file CLI.cpp
 * @brief Main entry point for the pptree command-line tool.
 *
 * Implements the train, predict, and evaluate subcommands including
 * data simulation, train/test splitting, progress display, model
 * serialization, and experiment export.
 */
#include "pptree.hpp"
#include "csv.hpp"

#include <fmt/format.h>
#include <cstdio>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <Eigen/Dense>
#include <filesystem>

#include "utils/Types.hpp"
#include <nlohmann/json.hpp>

#include "stats/DataPacket.hpp"
#include "stats/Normal.hpp"
#include "stats/Simulation.hpp"
#include "stats/ConfusionMatrix.hpp"

#include "cli/CLIOptions.hpp"
#include "cli/Presentation.hpp"
#include "cli/Color.hpp"
#include "io/IO.hpp"
#include "serialization/Json.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace pptree;
using namespace pptree::types;
using namespace pptree::stats;
using namespace pptree::cli;
using namespace pptree::io;
using namespace csv;
using json = nlohmann::json;

using pptree::variable_importance_permuted;
using pptree::variable_importance_projections;
using pptree::variable_importance_weighted_projections;

template<typename F>
auto measure_time_ms(F&& f) {
  auto str = std::chrono::high_resolution_clock::now();
  auto res = f();
  auto end = std::chrono::high_resolution_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - str);

  return std::make_pair(std::move(res), dur.count());
}


/**
 * @brief Display an in-place progress bar on stdout.
 * @param current   Current iteration (0-based before increment).
 * @param total     Total number of iterations.
 * @param quiet     If true, suppress all output.
 * @param bar_width Width of the progress bar in characters.
 */
void display_progress(int current, int total, bool quiet, int bar_width = 50) {
  if (quiet) return;

  float progress = static_cast<float>(current) / total;
  int pos        = static_cast<int>(bar_width * progress);

  std::string bar_template = emphasis("\r{} |");
  std::string bar          = std::string(pos, '-') + std::string(bar_width - pos, ' ');

  if (current == total) {
    bar_template = success(bar_template);
  } else {
    bar_template = info(bar_template);
  }

  fmt::print(bar_template + " {}/{} ({}%)     ", bar, current, total, static_cast<int>(progress * 100.0));
  std::fflush(stdout);

  if (current == total) {
    fmt::print("\n");
  }
}

/**
 * @brief Load or simulate data based on CLI options.
 *
 * If data_path is set, reads a CSV file; otherwise generates simulated data.
 * Ensures the response vector is contiguous (sorted by class).
 *
 * @param params The CLI options (data_path, simulate, sim_* parameters).
 * @param rng    Random number generator for simulation.
 * @return A DataPacket with features and labels.
 */
DataPacket read_data(const CLIOptions& params, pptree::stats::RNG& rng) {
  if (!params.data_path.empty()) {
    try {
      const DataPacket data = read_csv(params.data_path);

      FeatureMatrix x  = data.x;
      ResponseVector y = data.y;

      if (!GroupPartition::is_contiguous(y)) {
        pptree::stats::sort(x, y);
      }

      return DataPacket(x, y);
    } catch (const std::runtime_error& e) {
      fmt::print(stderr, "Error reading CSV file: {}\n", e.what());
      fmt::print(stderr, "Please ensure the file exists and is properly formatted\n");
      exit(1);
    } catch (const std::exception& e) {
      fmt::print(stderr, "Unexpected error reading file: {}\n", e.what());
      exit(1);
    }
  } else {
    SimulationParams simulation_params;
    simulation_params.mean            = params.sim_mean;
    simulation_params.mean_separation = params.sim_mean_separation;
    simulation_params.sd              = params.sim_sd;

    try {
      return simulate(params.rows, params.cols, params.classes, rng, simulation_params);
    } catch (const std::exception& e) {
      fmt::print(stderr, "Error simulating data: {}\n", e.what());
      exit(1);
    }
  }
}

/** @brief Result of a train operation containing the model and training duration. */
struct TrainResult {
  Model::Ptr model;
  long long duration;
};

/**
 * @brief Train a single model (Forest or Tree) on the full dataset.
 * @tparam Model Either Forest or Tree.
 * @param x      Feature matrix.
 * @param y      Response vector.
 * @param params CLI options (trees, seed, threads).
 * @param rng    Random number generator (used for single tree).
 * @return The trained model.
 */
TrainResult train_model(
  const FeatureMatrix&  x,
  const ResponseVector& y,
  CLIOptions const&     params,
  pptree::stats::RNG&   rng) {
  TrainingSpec::Ptr spec;
  std::function<Model::Ptr()> fact;

  if (params.trees > 0) {
    spec = TrainingSpecUGLDA::make(params.n_vars, params.lambda);
    fact = [&] {
      return std::make_unique<Forest>(Forest::train(*spec, x, y, params.trees, params.seed, params.threads));
    };
  } else {
    spec = TrainingSpecGLDA::make(params.lambda);
    fact = [&] {
      return std::make_unique<Tree>(Tree::train(*spec, x, y, rng));
    };
  }

  auto [model, ms] = measure_time_ms(fact);

  return { std::move(model), ms };
}

/** @brief Result of an evaluate run containing aggregated statistics. */
struct EvaluateResult {
  ModelStats stats;    ///< Per-iteration timing and error data.
};

/**
 * @brief Train and evaluate a model over multiple iterations.
 *
 * For each iteration, trains a fresh model on the training set,
 * records training time, train error, and test error. Displays
 * a progress bar unless quiet mode is active.
 *
 * @tparam Model Either Forest or Tree.
 * @param tr_x   Training features.
 * @param te_x   Test features.
 * @param tr_y   Training labels.
 * @param te_y   Test labels.
 * @param params CLI options (iterations, seed, threads, quiet).
 * @param rng    Random number generator.
 * @return An EvaluateResult with aggregated statistics.
 */
EvaluateResult evaluate_model(
  FeatureMatrix&      tr_x,
  FeatureMatrix&      te_x,
  ResponseVector&     tr_y,
  ResponseVector&     te_y,
  const CLIOptions&   params,
  pptree::stats::RNG& rng) {
  ModelStats model_stats;

  model_stats.tr_times = Vector<float>(params.iterations);
  model_stats.tr_error = Vector<float>(params.iterations);
  model_stats.te_error = Vector<float>(params.iterations);

  if (!params.quiet) {
    fmt::print("Running {} iterations:\n", emphasis(std::to_string(params.iterations)));
  }

  for (int i = 0; i < params.iterations; ++i) {
    display_progress(i, params.iterations, params.quiet);

    const auto train_result = train_model(tr_x, tr_y, params, rng);

    model_stats.tr_times[i] = train_result.duration;
    model_stats.tr_error[i] = pptree::stats::error_rate(train_result.model->predict(tr_x), tr_y);
    model_stats.te_error[i] = pptree::stats::error_rate(train_result.model->predict(te_x), te_y);

    display_progress(i + 1, params.iterations, params.quiet);
  }

  if (!params.quiet) {
    fmt::print("\n");
  }

  return { model_stats };
}

/**
 * @brief Save a trained model to a JSON file with config metadata.
 *
 * Produces a JSON file with structure: { model_type, model, config }.
 * The config block records all training parameters and the data path.
 *
 * @param model_json The serialized model JSON.
 * @param model_type "forest" or "tree".
 * @param params     CLI options for the config block.
 * @param path       Output file path.
 */
void save_model(
  const Model&       model,
  const CLIOptions&  params,
  const std::string& path) {
  json output = serialization::to_json(model);

  json config;
  config["trees"]   = params.trees;
  config["lambda"]  = params.lambda;
  config["seed"]    = params.seed;
  config["threads"] = params.threads;

  if (params.trees > 0 && params.n_vars > 0) {
    config["vars"] = params.n_vars;
  }

  if (!params.data_path.empty()) {
    config["data"] = params.data_path;
  }

  output["config"] = config;

  write_json_file(output, path);

  if (!params.quiet) {
    fmt::print("Model saved to {}\n", path);
  }
}

/**
 * @brief Load a model JSON file from disk.
 * @param path Path to the model JSON file.
 * @return The parsed JSON object.
 */
json load_model(const std::string& path) {
  std::ifstream in(path);

  if (!in.is_open()) {
    fmt::print(stderr, "Error: Could not open model file: {}\n", path);
    std::exit(1);
  }

  try {
    return json::parse(in);
  } catch (const json::parse_error& e) {
    fmt::print(stderr, "Error: Invalid JSON in model file: {}\n", e.what());
    std::exit(1);
  }
}

/**
 * @brief Build the JSON result for the predict subcommand.
 *
 * Always includes "predictions". When labels are available and
 * --no-metrics is not set, also includes "error_rate" and "confusion_matrix".
 *
 * @param predictions The predicted class labels.
 * @param data        The input dataset (may include actual labels).
 * @param no_metrics  If true, omit error_rate and confusion_matrix.
 * @return A JSON object with prediction results.
 */
json build_predict_result(
  const ResponseVector& predictions,
  const DataPacket&     data,
  bool                  no_metrics) {
  json result;
  std::vector<int> pred_vec(predictions.data(), predictions.data() + predictions.size());
  result["predictions"] = pred_vec;

  bool has_labels   = data.y.size() > 0;
  bool show_metrics = has_labels && !no_metrics;

  if (show_metrics) {
    ConfusionMatrix cm(predictions, data.y);
    result["error_rate"]       = cm.error();
    result["confusion_matrix"] = serialization::to_json(cm);
  }

  return result;
}

/**
 * @brief Build the config.json content for an experiment export bundle.
 *
 * Includes model parameters, evaluation parameters, and a relative
 * data path ("data.csv") for reproducibility.
 *
 * @param params The CLI options.
 * @return A JSON config object.
 */
json build_export_config(const CLIOptions& params) {
  json config;

  // Model parameters
  config["trees"]   = params.trees;
  config["lambda"]  = params.lambda;
  config["seed"]    = params.seed;
  config["threads"] = params.threads;

  if (params.trees > 0 && params.n_vars > 0) {
    config["vars"] = params.n_vars;
  }

  // Evaluation parameters
  config["train-ratio"] = params.train_ratio;
  config["iterations"]  = params.iterations;

  // Data source - point to local CSV for re-runnability
  config["data"] = "data.csv";

  return config;
}

/**
 * @brief Export a complete experiment bundle to a directory.
 *
 * Creates a directory containing:
 * - config.json: full training/evaluation configuration
 * - data.csv: the dataset used
 * - results.json: evaluation statistics with per-iteration breakdown
 *
 * @param params      CLI options (export_path, quiet).
 * @param full_data   The full dataset.
 * @param eval_result The evaluation results.
 */
void export_experiment(
  const CLIOptions&     params,
  const DataPacket&     full_data,
  const EvaluateResult& eval_result) {
  std::string dir = params.export_path;

  std::filesystem::create_directories(dir);

  // config.json
  json config = build_export_config(params);
  write_json_file(config, dir + "/config.json");

  // data.csv
  write_csv(full_data, dir + "/data.csv");

  // results.json
  write_json_file(eval_result.stats.to_json(), dir + "/results.json");

  if (!params.quiet) {
    fmt::print("{}{}/\n", success("Experiment exported to "), dir);
  }
}

int main(int argc, char *argv[]) {
  CLIOptions params = parse_args(argc, argv);

  init_color(params.no_color);

  // Post-parse: ensure .json extension on output paths
  if (!params.save_path.empty()) {
    params.save_path = ensure_json_extension(params.save_path);
  }

  if (!params.output_path.empty()) {
    params.output_path = ensure_json_extension(params.output_path);
  }

  #ifdef _OPENMP
  omp_set_num_threads(params.threads);
  #endif

  switch (params.subcommand) {
    case Subcommand::train: {
      // Validate save path before training
      if (!params.save_path.empty()) {
        check_file_not_exists(params.save_path);
      }

      pptree::stats::RNG rng(params.seed);
      auto data = read_data(params, rng);

      init_params(params, data.x.cols());
      announce_configuration(params);

      FeatureMatrix x  = data.x;
      ResponseVector y = data.y;

      const auto train_result = train_model(x, y, params, rng);

      if (!params.quiet) {
        fmt::print("Trained in {}ms\n", emphasis(std::to_string(train_result.duration)));
      }

      if (!params.save_path.empty()) {
        save_model(*train_result.model, params, params.save_path);
      }

      if (!params.no_metrics) {
        const auto *forest = dynamic_cast<const Forest *>(train_result.model.get());
        const auto *tree   = dynamic_cast<const Tree *>(train_result.model.get());
        FeatureVector vi1, vi2, vi3, scale;
        double oob_err = -1.0;

        const int n_vars = static_cast<int>(x.cols());

        scale = stats::sd(x);
        scale = (scale.array() > Feature(0)).select(scale, Feature(1));

        if (forest != nullptr) {
          oob_err = forest->oob_error(x, y);
          vi1 = variable_importance_permuted(*forest, x, y, params.seed);
          vi2 = variable_importance_projections(*forest, n_vars, &scale);
          vi3 = variable_importance_weighted_projections(*forest, x, y, &scale);
        } else if (tree != nullptr) {
          vi2 = variable_importance_projections(*tree, n_vars, &scale);
        }

        if (!params.quiet) {
          if (oob_err >= 0.0) {
            fmt::print("OOB error: {}\n", emphasis(fmt::format("{:.2f}%", oob_err * 100)));
          }

          print_variable_importance(vi1, vi2, vi3, scale);
        }

        if (!params.save_path.empty()) {
          std::ifstream in(params.save_path);
          json saved = json::parse(in);
          in.close();

          if (oob_err >= 0.0) {
            saved["oob_error"] = oob_err;
          }

          saved["variable_importance"] = vi_to_json(vi1, vi2, vi3, scale);
          write_json_file(saved, params.save_path);
        }
      }

      break;
    }

    case Subcommand::predict: {
      // Validate output path before doing work
      if (!params.output_path.empty()) {
        check_file_not_exists(params.output_path);
      }

      DataPacket data = [&]() {
        try {
          const DataPacket csv_data = read_csv(params.data_path);

          FeatureMatrix x  = csv_data.x;
          ResponseVector y = csv_data.y;

          if (!GroupPartition::is_contiguous(y)) {
            pptree::stats::sort(x, y);
          }

          return DataPacket(x, y);
        } catch (const std::exception& e) {
          fmt::print(stderr, "{} reading CSV file: {}\n", error("Error:"), e.what());
          std::exit(1);
        }
      }();

      json model_data  = load_model(params.model_path);
      auto model       = serialization::model_from_json(model_data);
      auto predictions = model->predict(data.x);

      bool has_labels   = data.y.size() > 0;
      bool show_metrics = has_labels && !params.no_metrics;

      // Terminal output: only metrics
      if (!params.quiet && show_metrics) {
        ConfusionMatrix cm(predictions, data.y);
        fmt::print("\n{}{:.2f}%\n\n", emphasis("Error rate: "), cm.error() * 100);
        print_confusion_matrix(cm);
      }

      // Hint about --output when not used
      if (!params.quiet && show_metrics && params.output_path.empty()) {
        fmt::print("{}\n", muted("Tip: use --output <file> to save individual predictions"));
      }

      // Save results to file if requested
      if (!params.output_path.empty()) {
        json file_result = build_predict_result(predictions, data, params.no_metrics);
        write_json_file(file_result, params.output_path);

        if (!params.quiet) {
          fmt::print("{}{}\n", success("Results saved to "), params.output_path);
        }
      }

      break;
    }

    case Subcommand::evaluate: {
      // Validate output paths before doing work
      if (!params.output_path.empty()) {
        check_file_not_exists(params.output_path);
      }

      if (!params.export_path.empty()) {
        check_dir_not_exists(params.export_path);
      }

      pptree::stats::RNG rng(params.seed);
      auto full_data = read_data(params, rng);

      init_params(params, full_data.x.cols());

      auto data_split = split(full_data, params.train_ratio, rng);

      FeatureMatrix tr_x  = full_data.x(data_split.tr, Eigen::all);
      FeatureMatrix te_x  = full_data.x(data_split.te, Eigen::all);
      ResponseVector tr_y = full_data.y(data_split.tr);
      ResponseVector te_y = full_data.y(data_split.te);

      announce_configuration(params, tr_x.rows(), te_x.rows());

      auto eval_result = evaluate_model(tr_x, te_x, tr_y, te_y, params, rng);

      // Measure peak RSS after evaluation
      eval_result.stats.peak_rss_bytes = get_peak_rss_bytes();

      if (!params.quiet) {
        announce_results(eval_result.stats);
      }

      // Save results to file if requested
      if (!params.output_path.empty()) {
        write_json_file(eval_result.stats.to_json(), params.output_path);

        if (!params.quiet) {
          fmt::print("{}{}\n", success("Results saved to "), params.output_path);
        }
      }

      // Export experiment bundle if requested
      if (!params.export_path.empty()) {
        export_experiment(params, full_data, eval_result);
      }

      break;
    }

    default:
      fmt::print(stderr, "Error: No subcommand specified\n");
      return 1;
  }

  return 0;
}
