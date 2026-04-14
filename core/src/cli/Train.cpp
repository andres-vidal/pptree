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
    sub->add_option("--stop", model.stop_input, "Stop rule strategy (e.g. pure_node)");
    sub->add_option("--binarize", model.binarize_input, "Binarization strategy (e.g. largest_gap)");
    sub->add_option("--partition", model.partition_input, "Partition strategy (e.g. by_group)");
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
    if (!params.data_path.empty()) {
      try {
        return io::csv::read_sorted(params.data_path);
      } catch (ppforest2::UserError const&) {
        throw;
      } catch (std::exception const& e) {
        throw ppforest2::UserError(fmt::format("Error reading CSV file '{}': {}", params.data_path, e.what()));
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

  TrainResult
  train_model(FeatureMatrix const& x, OutcomeVector const& y, Params const& params, ppforest2::stats::RNG& rng) {
    auto const& m = params.model;

    auto spec = TrainingSpec::from_json({
        {"pp", m.pp_config},
        {"vars", m.vars_config},
        {"cutpoint", m.cutpoint_config},
        {"stop", m.stop_config},
        {"binarize", m.binarize_config},
        {"partition", m.partition_config},
        {"leaf", m.leaf_config},
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

    FeatureMatrix const x = data.x;
    OutcomeVector const y = data.y;

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
