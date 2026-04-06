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
  void add_model_options(CLI::App* sub, ModelParams& model) {
    sub->add_option("-n,--size", model.size, "Number of trees (default: 100, 0 for single tree)")
        ->check(CLI::NonNegativeNumber);
    auto* lambda_opt = sub->add_option("-l,--lambda", model.lambda, "Method selection (0=LDA, (0,1]=PDA)")
                           ->check(CLI::Range(0.0F, 1.0F));
    sub->add_option("--threads", model.threads, "Number of threads (default: CPU cores)")->check(CLI::PositiveNumber);
    sub->add_option("-r,--seed", model.seed, "Random seed (default: random)");
    auto* n_vars_opt =
        sub->add_option("--n-vars", model.n_vars, "Features per split (integer count)")->check(CLI::PositiveNumber);
    auto* p_vars_opt = sub->add_option(
        "--p-vars", model.p_vars_input, "Features per split (proportion, e.g. 0.5 or 1/2, default: 0.5)"
    );
    sub->add_option("--max-retries", model.max_retries, "Max retries for degenerate trees (default: 3)")
        ->check(CLI::NonNegativeNumber);

    n_vars_opt->excludes(p_vars_opt);
    p_vars_opt->excludes(n_vars_opt);

    // Explicit strategy flags (mutually exclusive with shortcut params)
    auto* pp_opt = sub->add_option("--pp", model.pp_input, "PP strategy (e.g. pda, pda:lambda=0.5)");
    auto* vars_opt =
        sub->add_option("--vars", model.vars_input, "Variable selection strategy (e.g. all, uniform:count=3)");
    sub->add_option("--cutpoint", model.cutpoint_input, "Split cutpoint strategy (e.g. mean_of_means)");
    sub->add_option("--stop", model.stop_input, "Stop rule strategy (e.g. pure_node)");
    sub->add_option("--binarize", model.binarize_input, "Binarization strategy (e.g. largest_gap)");
    sub->add_option("--partition", model.partition_input, "Partition strategy (e.g. by_group)");
    sub->add_option("--leaf", model.leaf_input, "Leaf strategy (e.g. majority_vote)");

    pp_opt->excludes(lambda_opt);
    lambda_opt->excludes(pp_opt);
    vars_opt->excludes(n_vars_opt);
    vars_opt->excludes(p_vars_opt);
    n_vars_opt->excludes(vars_opt);
    p_vars_opt->excludes(vars_opt);
  }

  CLI::App* setup_train(CLI::App& app, CLIOptions& params) {
    auto* sub = app.add_subcommand("train", "Train a model");
    sub->add_option("-d,--data", params.data_path, "CSV training data")->required()->check(CLI::ExistingFile);
    add_model_options(sub, params.model);
    auto* save_opt =
        sub->add_option("-s,--save", params.save_path, "Save trained model to JSON file (default: model.json)");
    auto* no_save = sub->add_flag("--no-save", params.no_save, "Skip saving the model (for benchmarking)");
    save_opt->excludes(no_save);
    no_save->excludes(save_opt);
    sub->add_flag("--no-metrics", params.no_metrics, "Skip variable importance computation and output");
    return sub;
  }
}

namespace ppforest2::cli {
  DataPacket read_data(CLIOptions const& params, ppforest2::stats::RNG& rng) {
    if (!params.data_path.empty()) {
      try {
        return io::csv::read_sorted(params.data_path);
      } catch (ppforest2::UserError const& e) {
        fmt::print(stderr, "Error: {}\n", e.what());
        fmt::print(stderr, "File: {}\n", params.data_path);
        exit(1);
      } catch (std::runtime_error const& e) {
        fmt::print(stderr, "Error reading CSV file: {}\n", e.what());
        fmt::print(stderr, "Please ensure the file exists and is properly formatted\n");
        exit(1);
      } catch (std::exception const& e) {
        fmt::print(stderr, "Unexpected error reading file: {}\n", e.what());
        exit(1);
      }
    } else {
      SimulationParams simulation_params;
      simulation_params.mean            = params.simulation.mean;
      simulation_params.mean_separation = params.simulation.mean_separation;
      simulation_params.sd              = params.simulation.sd;

      try {
        return simulate(
            params.simulation.rows, params.simulation.cols, params.simulation.n_groups, rng, simulation_params
        );
      } catch (std::exception const& e) {
        fmt::print(stderr, "Error simulating data: {}\n", e.what());
        exit(1);
      }
    }
  }

  TrainResult
  train_model(FeatureMatrix const& x, OutcomeVector const& y, CLIOptions const& params, ppforest2::stats::RNG& rng) {
    auto const& m     = params.model;
    auto default_json = [](nlohmann::json const& config, std::string const& default_name) {
      return config.is_null() ? json{{"name", default_name}} : config;
    };

    json const spec_json = {
        {"pp", m.pp_config},
        {"vars", m.vars_config},
        {"cutpoint", m.cutpoint_config},
        {"stop", default_json(m.stop_config, "pure_node")},
        {"binarize", default_json(m.binarize_config, "largest_gap")},
        {"partition", default_json(m.partition_config, "by_group")},
        {"leaf", default_json(m.leaf_config, "majority_vote")},
        {"size", m.size},
        {"seed", m.seed},
        {"threads", m.threads},
        {"max_retries", m.max_retries},
    };

    auto spec = TrainingSpec::from_json(spec_json);

    auto [model, ms] = io::measure_time_ms([&] { return Model::train(*spec, x, y); });

    return {std::move(model), ms};
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

    FeatureMatrix const x  = data.x;
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
