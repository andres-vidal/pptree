/**
 * @file Train.cpp
 * @brief Model training utilities and train subcommand handler.
 */
#include "cli/Train.hpp"
#include "cli/Validation.hpp"

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include "stats/Simulation.hpp"
#include "io/Presentation.hpp"
#include "io/Output.hpp"
#include "io/IO.hpp"
#include "io/Timing.hpp"
#include "serialization/Json.hpp"
#include "utils/UserError.hpp"
#include "utils/Types.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using json = nlohmann::json;

namespace ppforest2::cli {
  void add_model_options(CLI::App* sub, ModelParams& model) {
    // `mode_input` defaults to "classification" in the ModelParams struct.
    // We don't use CLI11's default_val here because it would override config-file
    // values whenever the CLI flag isn't passed.
    sub->add_option("-m,--mode", model.mode_input, "Training mode (classification or regression)");
    sub->add_option("-n,--size", model.size, "Number of trees (default: 100, 0 for single tree)");
    sub->add_option("-l,--lambda", model.lambda, "Method selection (0=LDA, (0,1]=PDA)");
    sub->add_option("--threads", model.threads, "Number of threads (default: CPU cores)");
    sub->add_option("-r,--seed", model.seed, "Random seed (default: random)");
    sub->add_option("--n-vars", model.n_vars, "Features per split (integer count)");
    sub->add_option("--p-vars", model.p_vars_input, "Features per split (proportion, e.g. 0.5 or 1/2, default: 0.5)");
    sub->add_option("--max-retries", model.max_retries, "Max retries for degenerate trees (default: 3)");

    sub->add_option("--pp", model.pp_input, "PP strategy (e.g. pda, pda:lambda=0.5)");
    sub->add_option("--vars", model.vars_input, "Variable selection strategy (e.g. all, uniform:count=3)");
    sub->add_option("--cutpoint", model.cutpoint_input, "Split cutpoint strategy (e.g. mean_of_means)");
    // `--stop` is repeatable: each occurrence is one rule, and two or more
    // occurrences are wrapped in `any(...)` at resolve time. A single
    // `--stop` keeps the prior flat-syntax behaviour. For the full
    // nested-JSON shape (rare — only when you need something other than
    // `any` composition, which the library doesn't currently ship),
    // use `--config <file>`.
    sub->add_option(
        "--stop", model.stop_inputs,
        "Stop rule strategy (e.g. pure_node, min_size:min_size=5). "
        "Repeat to compose rules via any(...) — e.g. "
        "--stop min_size:min_size=5 --stop min_variance:threshold=0.01."
    );
    sub->add_option("--binarize", model.binarize_input, "Binarization strategy (e.g. largest_gap)");
    sub->add_option("--grouping", model.grouping_input, "Grouping strategy (e.g. by_label)");
    sub->add_option("--leaf", model.leaf_input, "Leaf strategy (e.g. majority_vote)");

    // Mutual exclusions (all configurable via config, so no CLI11 validators)
    sub->get_option("--n-vars")->excludes("--p-vars");
    sub->get_option("--p-vars")->excludes("--n-vars");
    sub->get_option("--pp")->excludes("--lambda");
    sub->get_option("--lambda")->excludes("--pp");
    sub->get_option("--vars")->excludes("--n-vars");
    sub->get_option("--vars")->excludes("--p-vars");
    sub->get_option("--n-vars")->excludes("--vars");
    sub->get_option("--p-vars")->excludes("--vars");
  }

  void setup_train(CLI::App& app, Params& params) {
    auto* sub = app.add_subcommand("train", "Train a model");
    sub->add_option("-d,--data", params.data_path, "CSV training data");

    add_model_options(sub, params.model);

    // CLI-exclusive options
    sub->add_option("-s,--save", params.save_path, "Save trained model to JSON file (default: model.json)");
    sub->add_flag("--no-save", params.no_save, "Skip saving the model (for benchmarking)");
    sub->add_flag("--no-metrics", params.no_metrics, "Skip variable importance computation and output");

    // CLI-exclusive constraints
    sub->get_option("--save")->excludes("--no-save");
    sub->get_option("--no-save")->excludes("--save");

    sub->callback([&]() { params.subcommand = Subcommand::train; });
  }
}

namespace ppforest2::cli {
  DataPacket read_data(Params const& params, ppforest2::stats::RNG& rng) {
    bool const is_regression = params.model.mode_input == "regression";

    if (!params.data_path.empty()) {
      try {
        return is_regression ? io::csv::read_regression_sorted(params.data_path)
                              : io::csv::read_sorted(params.data_path);
      } catch (ppforest2::UserError const&) {
        throw;
      } catch (std::exception const& e) {
        throw ppforest2::UserError(fmt::format("Error reading CSV file '{}': {}", params.data_path, e.what()));
      }
    }

    if (is_regression) {
      RegressionSimulationParams reg_params;
      // Use the simulation sd as feature_sd; noise_sd defaults to 0.1.
      reg_params.feature_sd = params.simulation.sd;

      try {
        return simulate_regression(params.simulation.rows, params.simulation.cols, rng, reg_params);
      } catch (std::exception const& e) {
        throw ppforest2::UserError(fmt::format("Error simulating regression data: {}", e.what()));
      }
    }

    SimulationParams simulation_params;
    simulation_params.mean            = params.simulation.mean;
    simulation_params.mean_separation = params.simulation.mean_separation;
    simulation_params.sd              = params.simulation.sd;

    try {
      return simulate(
          params.simulation.rows, params.simulation.cols, params.simulation.n_groups, rng, simulation_params
      );
    } catch (std::exception const& e) {
      throw ppforest2::UserError(fmt::format("Error simulating data: {}", e.what()));
    }
  }

  TrainResult train_model(
      FeatureMatrix& x,
      OutcomeVector& y,
      Params const& params,
      ppforest2::stats::RNG& rng
  ) {
    auto const& m = params.model;

    auto spec = TrainingSpec::from_json({
        {"pp", m.pp_config},
        {"vars", m.vars_config},
        {"cutpoint", m.cutpoint_config},
        {"stop", m.stop_config},
        {"binarize", m.binarize_config},
        {"grouping", m.grouping_config},
        {"leaf", m.leaf_config},
        {"mode", m.mode_input},
        {"size", m.size},
        {"seed", *m.seed},
        {"threads", *m.threads},
        {"max_retries", m.max_retries},
    });

    auto [model, ms] = io::measure_time_ms([&] { return Model::train(*spec, x, y); });

    return {std::move(model), ms};
  }

  int run_train(Params& params) {
    // Snapshot which params are unset before defaults are filled
    bool default_seed    = !params.model.seed;
    bool default_threads = !params.model.threads;
    bool default_vars    = !params.model.p_vars && !params.model.n_vars;

    params.evaluate.resolve_defaults();
    params.resolve_seed();
    validate_params(params);

    if (params.no_save) {
      params.save_path.clear();
    }

    io::Output out(params.quiet);
    warn_unused_params(out, params);

    // Validate save path before training
    if (!params.save_path.empty()) {
      io::check_file_not_exists(params.save_path);
    }

    ppforest2::stats::RNG rng(*params.model.seed);
    auto data = read_data(params, rng);

    params.resolve_defaults(data.x.cols());

    // `data.x` and `data.y` may be permuted in place during regression
    // training (ByCutpoint). That's fine: `data` is a transient carrier —
    // the CLI discards it after `run_train` returns, and metrics computed
    // on the permuted layout are still correct because `y` is permuted in
    // lockstep (per-observation pairings are preserved).
    auto train_result = train_model(data.x, data.y, params, rng);

    serialization::Export<Model::Ptr> model_export{
        std::move(train_result.model),
        data.group_names,
        nullptr,
        static_cast<int>(data.x.rows()),
        static_cast<int>(data.x.cols()),
        data.feature_names,
    };

    if (!params.no_metrics) {
      model_export.compute_metrics(data.x, data.y);
    }

    json model_json = model_export.to_json();

    if (!params.data_path.empty()) {
      model_json["config"]["data"] = params.data_path;
    }

    model_json["training_duration_ms"] = train_result.duration;

    // Save model
    if (!params.save_path.empty()) {
      io::json::write_file(model_json, params.save_path, user_error);
      model_json["save_path"] = params.save_path;
    }

    io::ConfigDisplayHints hints;
    hints.vars_percent    = params.model.size > 0 && params.model.p_vars ? *params.model.p_vars * 100 : -1;
    hints.default_vars    = default_vars;
    hints.default_threads = default_threads;
    hints.default_seed    = default_seed;

    io::print_summary(out, model_json, hints);

    return 0;
  }
}
