/**
 * @file Train.cpp
 * @brief Model training utilities and train subcommand handler.
 */
#include "cli/Train.hpp"
#include "ppforest2.hpp"

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <fstream>

#include "stats/Simulation.hpp"
#include "io/Presentation.hpp"
#include "io/Output.hpp"
#include "io/IO.hpp"
#include "io/Timing.hpp"
#include "serialization/Json.hpp"
#include "utils/UserError.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using json = nlohmann::json;

namespace ppforest2::cli {
  void add_model_options(CLI::App *sub, ModelParams& model) {
    sub->add_option("-n,--size", model.size, "Number of trees (default: 100, 0 for single tree)")
    ->check(CLI::NonNegativeNumber);
    auto lambda_opt = sub->add_option("-l,--lambda", model.lambda, "Method selection (0=LDA, (0,1]=PDA)")
      ->check(CLI::Range(0.0f, 1.0f));
    sub->add_option("--threads", model.threads, "Number of threads (default: CPU cores)")
    ->check(CLI::PositiveNumber);
    sub->add_option("-r,--seed", model.seed, "Random seed (default: random)");
    auto vars_opt = sub->add_option("-v,--vars", model.vars_input, "Features per split (integer=count, decimal or fraction=proportion, default: 0.5)");
    sub->add_option("--max-retries", model.max_retries, "Max retries for degenerate trees (default: 3)")
    ->check(CLI::NonNegativeNumber);

    // Explicit strategy flags (mutually exclusive with shortcut params)
    auto pp_opt = sub->add_option("--pp", model.pp_input, "PP strategy (e.g. pda, pda:lambda=0.5)");
    auto dr_opt = sub->add_option("--dr", model.dr_input, "DR strategy (e.g. noop, uniform:vars=3)");
    sub->add_option("--sr", model.sr_input, "Split rule strategy (e.g. mean_of_means)");

    pp_opt->excludes(lambda_opt);
    lambda_opt->excludes(pp_opt);
    dr_opt->excludes(vars_opt);
    vars_opt->excludes(dr_opt);
  }

  CLI::App * setup_train(CLI::App& app, CLIOptions& params) {
    auto sub = app.add_subcommand("train", "Train a model");
    sub->add_option("-d,--data", params.data_path, "CSV training data")
    ->required()
    ->check(CLI::ExistingFile);
    add_model_options(sub, params.model);
    auto save_opt = sub->add_option("-s,--save", params.save_path, "Save trained model to JSON file (default: model.json)");
    auto no_save  = sub->add_flag("--no-save", params.no_save, "Skip saving the model (for benchmarking)");
    save_opt->excludes(no_save);
    no_save->excludes(save_opt);
    sub->add_flag("--no-metrics", params.no_metrics, "Skip variable importance computation and output");
    return sub;
  }
}

namespace ppforest2::cli {
  DataPacket read_data(const CLIOptions& params, ppforest2::stats::RNG& rng) {
    if (!params.data_path.empty()) {
      try {
        return io::csv::read_sorted(params.data_path);
      } catch (const ppforest2::UserError& e) {
        fmt::print(stderr, "Error: {}\n", e.what());
        fmt::print(stderr, "File: {}\n", params.data_path);
        exit(1);
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
      simulation_params.mean            = params.simulation.mean;
      simulation_params.mean_separation = params.simulation.mean_separation;
      simulation_params.sd              = params.simulation.sd;

      try {
        return simulate(params.simulation.rows, params.simulation.cols, params.simulation.n_groups, rng, simulation_params);
      } catch (const std::exception& e) {
        fmt::print(stderr, "Error simulating data: {}\n", e.what());
        exit(1);
      }
    }
  }

  TrainResult train_model(
    const FeatureMatrix&   x,
    const ResponseVector&  y,
    const CLIOptions&      params,
    ppforest2::stats::RNG& rng) {
    auto& m   = params.model;
    auto spec = TrainingSpec::from_json({
      { "pp",          m.pp_config },
      { "dr",          m.dr_config },
      { "sr",          m.sr_config },
      { "size",        m.size },
      { "seed",        m.seed },
      { "threads",     m.threads },
      { "max_retries", m.max_retries },
    });

    auto [model, ms] = io::measure_time_ms([&] {
          return Model::train(*spec, x, y);
        });

    return { std::move(model), ms };
  }

  int run_train(CLIOptions& params) {
    io::Output out(params.quiet);

    // Validate save path before training
    if (!params.save_path.empty()) {
      io::check_file_not_exists(params.save_path);
    }

    ppforest2::stats::RNG rng(params.model.seed);
    auto data = read_data(params, rng);

    init_params(params, data.x.cols());

    FeatureMatrix x  = data.x;
    ResponseVector y = data.y;

    auto train_result = train_model(x, y, params, rng);

    serialization::Export<Model::Ptr> model_export{
      std::move(train_result.model),
      data.group_names,
      nullptr,
      static_cast<int>(x.rows()),
      static_cast<int>(x.cols()),
      data.feature_names,
    };

    if (!params.no_metrics) {
      model_export.compute_metrics(x, y);
    }

    json model_json = model_export.to_json();

    if (!params.data_path.empty()) {
      model_json["config"]["data"] = params.data_path;
    }

    model_json["training_duration_ms"] = train_result.duration;

    // Save model
    if (!params.save_path.empty()) {
      io::json::write_file(model_json, params.save_path);
      model_json["save_path"] = params.save_path;
    }

    io::ConfigDisplayHints hints;
    hints.vars_percent    = params.model.size > 0 ? static_cast<int>(params.model.p_vars * 100) : -1;
    hints.default_vars    = params.model.used_default_vars;
    hints.default_threads = params.model.used_default_threads;
    hints.default_seed    = params.model.used_default_seed;

    io::print_summary(out, model_json, hints);

    return 0;
  }
}
