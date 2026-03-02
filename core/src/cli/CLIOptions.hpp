/**
 * @file CLIOptions.hpp
 * @brief CLI argument parsing, validation, and configuration for pptree.
 *
 * Defines the CLIOptions struct, the ConfigJSON adapter for CLI11,
 * and functions to parse, validate, and initialize runtime parameters.
 */
#pragma once

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <random>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef PPTREE_VERSION
#define PPTREE_VERSION "0.0.0"
#endif

namespace pptree::cli {
  /**
   * @brief JSON config file adapter for CLI11.
   *
   * Reads and writes CLI11 options from/to JSON files.
   * Supports broadcasting top-level keys to all subcommands
   * listed in `subcommand_names_`.
   */
  class ConfigJSON : public CLI::Config {
    public:
      ConfigJSON(std::vector<std::string> subcommand_names = {})
        : subcommand_names_(std::move(subcommand_names)) {
      }

      std::string to_config(const CLI::App *app, bool default_also, bool, std::string) const override {
        nlohmann::json j;

        for (const CLI::Option *opt : app->get_options({})) {
          if (!opt->get_lnames().empty() && opt->get_configurable()) {
            std::string name = opt->get_lnames()[0];

            if (opt->get_type_size() != 0) {
              if (opt->count() == 1) j[name] = opt->results().at(0);
              else if (opt->count() > 1) j[name] = opt->results();
              else if (default_also && !opt->get_default_str().empty()) j[name] = opt->get_default_str();
            } else if (opt->count() == 1) {
              j[name] = true;
            } else if (opt->count() > 1) {
              j[name] = opt->count();
            } else if (opt->count() == 0 && default_also) {
              j[name] = false;
            }
          }
        }

        for (const CLI::App *subcom : app->get_subcommands({})) {
          j[subcom->get_name()] = nlohmann::json(to_config(subcom, default_also, false, ""));
        }

        return j.dump(4);
      }

      std::vector<CLI::ConfigItem> from_config(std::istream &input) const override {
        nlohmann::json j;
        input >> j;
        return _from_config(j);
      }

    private:
      std::vector<std::string> subcommand_names_;

      std::vector<CLI::ConfigItem>
      _from_config(nlohmann::json j, std::string name = "", std::vector<std::string> prefix = {}) const {
        std::vector<CLI::ConfigItem> results;

        if (j.is_object()) {
          for (auto item = j.begin(); item != j.end(); ++item) {
            auto copy_prefix = prefix;

            if (!name.empty()) copy_prefix.push_back(name);

            auto sub_results = _from_config(*item, item.key(), copy_prefix);
            results.insert(results.end(), sub_results.begin(), sub_results.end());
          }
        } else if (!name.empty()) {
          if (prefix.empty() && !subcommand_names_.empty()) {
            for (const auto& sub_name : subcommand_names_) {
              results.emplace_back();
              CLI::ConfigItem &res = results.back();
              res.name    = name;
              res.parents = { sub_name };
              _set_inputs(res, j);
            }
          } else {
            results.emplace_back();
            CLI::ConfigItem &res = results.back();
            res.name    = name;
            res.parents = prefix;
            _set_inputs(res, j);
          }
        }

        return results;
      }

      void _set_inputs(CLI::ConfigItem &res, const nlohmann::json &j) const {
        if (j.is_boolean()) {
          res.inputs = { j.get<bool>() ? "true" : "false" };
        } else if (j.is_number()) {
          std::stringstream ss;
          ss << j.get<double>();
          res.inputs = { ss.str() };
        } else if (j.is_string()) {
          res.inputs = { j.get<std::string>() };
        } else if (j.is_array()) {
          for (const auto &val : j) {
            res.inputs.push_back(val.get<std::string>());
          }
        } else {
          throw CLI::ConversionError("Failed to convert " + res.name);
        }
      }
  };
  /** @brief Available CLI subcommands. */
  enum class Subcommand { none, train, predict, evaluate };

  /**
   * @brief All CLI options and runtime parameters.
   *
   * Fields with -1 or empty defaults are sentinel values meaning
   * "not set by the user" and will be resolved by init_params().
   */
  struct CLIOptions {
    int trees    = 100;
    float lambda = 0.5;
    int threads  = -1;
    int seed     = -1;
    float p_vars = -1;
    int n_vars   = -1;
    std::string vars_input;
    float train_ratio = 0.7;
    int iterations    = 1;
    std::string data_path;
    std::string simulate;
    int rows                  = 1000;
    int cols                  = 10;
    int classes               = 2;
    float sim_mean            = 100.0f;
    float sim_mean_separation = 50.0f;
    float sim_sd              = 10.0f;

    Subcommand subcommand = Subcommand::none;
    std::string save_path = "model.json";
    std::string model_path;
    std::string output_path;
    std::string export_path;
    bool quiet      = false;
    bool no_save    = false;
    bool no_metrics = false;
    bool no_color   = false;

    bool used_default_seed    = false;
    bool used_default_threads = false;
    bool used_default_vars    = false;
  };

  /**
   * @brief Warn the user about parameters that are ignored for single-tree training.
   *
   * When trees==0 (single tree mode), threads and vars have no effect.
   * Prints warnings to stdout unless quiet mode is active.
   *
   * @param params The CLI options to check.
   */
  inline void warn_unused_params(const CLIOptions& params) {
    if (params.quiet) return;

    if (params.trees == 0) {
      bool has_warnings = false;

      if (params.threads != -1) {
        fmt::print("Warning: threads parameter is ignored when training a single tree\n");
        has_warnings = true;
      }

      if (params.p_vars != -1 || params.n_vars != -1) {
        fmt::print("Warning: --vars parameter is ignored when training a single tree\n");
        has_warnings = true;
      }

      if (has_warnings) {
        fmt::print("Single trees always use all features for splitting\n");
      }
    }
  }

  /**
   * @brief Resolve sentinel values in CLIOptions to concrete defaults.
   *
   * - lambda: defaults to 0.5 if -1.
   * - seed: randomly generated if -1.
   * - threads: set to max OpenMP threads if -1.
   * - vars: computed from total_vars if neither p_vars nor n_vars was set.
   *
   * Also sets the `used_default_*` tracking flags.
   *
   * @param params     The CLI options to initialize (modified in place).
   * @param total_vars Total number of feature columns (0 to skip vars resolution).
   */
  inline void init_params(CLIOptions& params, int total_vars = 0) {
    if (params.lambda == -1) {
      params.lambda = 0.5;
    }

    if (params.train_ratio <= 0 || params.train_ratio >= 1) {
      fmt::print(stderr, "Error: Train ratio must be between 0 and 1\n");
      exit(1);
    }

    if (params.seed == -1) {
      std::random_device rd;
      params.seed              = rd();
      params.used_default_seed = true;
    }

    if (params.threads == -1) {
      #ifdef _OPENMP
      params.threads = omp_get_max_threads();
      #else
      params.threads = 1;
      #endif
      params.used_default_threads = true;
    }

    if (total_vars > 0 && params.trees > 0) {
      if (params.n_vars != -1) {
        params.p_vars = static_cast<float>(params.n_vars) / total_vars;
      } else if (params.p_vars != -1) {
        params.n_vars = std::round(total_vars * params.p_vars);
      } else {
        params.p_vars            = 0.5;
        params.n_vars            = std::round(total_vars * params.p_vars);
        params.used_default_vars = true;
      }
    }
  }

  /**
   * @brief Parse command-line arguments into a CLIOptions struct.
   *
   * Uses CLI11 to define and parse three subcommands (train, predict, evaluate)
   * plus global options (--quiet, --no-color, --config, --version).
   * Performs post-parse validation on --vars, --simulate format, and
   * mutually exclusive options. Exits on parse errors.
   *
   * @param argc Argument count from main().
   * @param argv Argument vector from main().
   * @return A populated CLIOptions struct.
   */
  inline CLIOptions parse_args(int argc, char *argv[]) {
    CLIOptions params;

    CLI::App app{ "pptree - Projection Pursuit Trees and Forests" };
    app.require_subcommand(1);
    app.set_version_flag("--version,-V", PPTREE_VERSION, "Print version and exit");

    // Global options
    app.add_flag("--quiet,-q", params.quiet, "Suppress all terminal output");
    app.add_flag("--no-color", params.no_color, "Disable colored output");
    app.config_formatter(std::make_shared<ConfigJSON>(
        std::vector<std::string>{ "train", "predict", "evaluate" }));
    app.set_config("--config", "", "Read parameters from JSON config file");

    // Helper: add model training options shared by train and evaluate
    auto add_model_options = [&](CLI::App *sub) {
        sub->add_option("-t,--trees", params.trees, "Number of trees (default: 100, 0 for single tree)")
        ->check(CLI::NonNegativeNumber);
        sub->add_option("-l,--lambda", params.lambda, "Method selection (0=LDA, (0,1]=PDA)")
        ->check(CLI::Range(0.0f, 1.0f));
        sub->add_option("--threads", params.threads, "Number of threads (default: CPU cores)")
        ->check(CLI::PositiveNumber);
        sub->add_option("-r,--seed", params.seed, "Random seed (default: random)");
        sub->add_option("-v,--vars", params.vars_input, "Features per split (integer=count, decimal or fraction=proportion, default: 0.5)");
      };

    // Train subcommand
    auto train_sub = app.add_subcommand("train", "Train a model");
    train_sub->add_option("-d,--data", params.data_path, "CSV training data")
    ->required()
    ->check(CLI::ExistingFile);
    add_model_options(train_sub);
    auto train_save_opt = train_sub->add_option("-s,--save", params.save_path, "Save trained model to JSON file (default: model.json)");
    auto train_no_save  = train_sub->add_flag("--no-save", params.no_save, "Skip saving the model (for benchmarking)");
    train_save_opt->excludes(train_no_save);
    train_no_save->excludes(train_save_opt);

    // Predict subcommand
    auto predict_sub = app.add_subcommand("predict", "Load a model and predict on new data");
    predict_sub->add_option("-M,--model", params.model_path, "Saved model JSON file")
    ->required()
    ->check(CLI::ExistingFile);
    predict_sub->add_option("-d,--data", params.data_path, "CSV data to predict on")
    ->required()
    ->check(CLI::ExistingFile);
    predict_sub->add_option("-o,--output", params.output_path, "Save prediction results to JSON file");
    predict_sub->add_flag("--no-metrics", params.no_metrics, "Omit error rate and confusion matrix from output");

    // Evaluate subcommand
    auto eval_sub      = app.add_subcommand("evaluate", "Train and evaluate a model");
    auto eval_data_opt = eval_sub->add_option("-d,--data", params.data_path, "CSV file")
      ->check(CLI::ExistingFile);
    auto eval_sim_opt = eval_sub->add_option("--simulate", params.simulate, "Simulate NxMxK data");
    eval_data_opt->excludes(eval_sim_opt);
    eval_sim_opt->excludes(eval_data_opt);

    add_model_options(eval_sub);
    eval_sub->add_option("-p,--train-ratio", params.train_ratio, "Train set ratio (default: 0.7)")
    ->check(CLI::Range(0.01f, 0.99f));
    eval_sub->add_option("-i,--iterations", params.iterations, "Number of training iterations (default: 1)")
    ->check(CLI::PositiveNumber);
    eval_sub->add_option("--sim-mean", params.sim_mean, "Mean for simulated data (default: 100.0)")
    ->needs(eval_sim_opt);
    eval_sub->add_option("--sim-mean-separation", params.sim_mean_separation, "Mean separation between classes (default: 50.0)")
    ->needs(eval_sim_opt)
    ->check(CLI::PositiveNumber);
    eval_sub->add_option("--sim-sd", params.sim_sd, "Standard deviation for simulated data (default: 10.0)")
    ->needs(eval_sim_opt)
    ->check(CLI::PositiveNumber);
    eval_sub->add_option("-o,--output", params.output_path, "Save evaluation results to JSON file");
    eval_sub->add_option("-e,--export", params.export_path, "Export experiment bundle to directory");

    // Parse
    try {
      app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
      std::exit(app.exit(e));
    }

    // Post-parse: determine subcommand
    if (train_sub->parsed()) {
      params.subcommand = Subcommand::train;
    } else if (predict_sub->parsed()) {
      params.subcommand = Subcommand::predict;
    } else if (eval_sub->parsed()) {
      params.subcommand = Subcommand::evaluate;
    }

    // Post-parse: handle --no-save for train
    if (params.subcommand == Subcommand::train && params.no_save) {
      params.save_path.clear();
    }

    // Post-parse: evaluate has no --save
    if (params.subcommand == Subcommand::evaluate) {
      params.save_path.clear();
    }

    // Post-parse: interpret --vars input
    if (!params.vars_input.empty()) {
      try {
        auto slash_pos = params.vars_input.find('/');

        if (slash_pos != std::string::npos) {
          int numerator   = std::stoi(params.vars_input.substr(0, slash_pos));
          int denominator = std::stoi(params.vars_input.substr(slash_pos + 1));

          if (denominator <= 0) {
            fmt::print(stderr, "Error: --vars fraction denominator must be positive\n");
            std::exit(1);
          }

          if (numerator <= 0) {
            fmt::print(stderr, "Error: --vars fraction numerator must be positive\n");
            std::exit(1);
          }

          float val = static_cast<float>(numerator) / static_cast<float>(denominator);

          if (val > 1) {
            fmt::print(stderr, "Error: --vars fraction must evaluate to a proportion between 0 and 1\n");
            std::exit(1);
          }

          params.p_vars = val;
        } else if (params.vars_input.find('.') != std::string::npos) {
          float val = std::stof(params.vars_input);

          if (val <= 0 || val > 1) {
            fmt::print(stderr, "Error: --vars proportion must be between 0 and 1\n");
            std::exit(1);
          }

          params.p_vars = val;
        } else {
          int val = std::stoi(params.vars_input);

          if (val <= 0) {
            fmt::print(stderr, "Error: --vars count must be positive\n");
            std::exit(1);
          }

          params.n_vars = val;
        }
      } catch (const std::exception&) {
        fmt::print(stderr, "Error: Invalid --vars value: {}\n", params.vars_input);
        std::exit(1);
      }
    }

    // Post-parse: validate simulate format
    if (!params.simulate.empty()) {
      std::string sim_str = params.simulate;
      size_t x1           = sim_str.find('x');
      size_t x2           = sim_str.find('x', x1 + 1);

      if (x1 == std::string::npos || x2 == std::string::npos) {
        fmt::print(stderr, "Error: Simulate format must be NxMxK (e.g., 1000x10x2)\n");
        std::exit(1);
      }

      try {
        params.rows    = std::stoi(sim_str.substr(0, x1));
        params.cols    = std::stoi(sim_str.substr(x1 + 1, x2 - x1 - 1));
        params.classes = std::stoi(sim_str.substr(x2 + 1));

        if (params.rows <= 0 || params.cols <= 0 || params.classes <= 1) {
          throw std::out_of_range("Values must be positive and classes must be > 1");
        }
      } catch (const std::exception& e) {
        fmt::print(stderr, "Error: Invalid simulate values: {}\n", e.what());
        std::exit(1);
      }
    }

    // Post-parse: evaluate requires data source
    if (params.subcommand == Subcommand::evaluate && params.simulate.empty() && params.data_path.empty()) {
      fmt::print(stderr, "Error: Must specify either --simulate or --data\n");
      std::exit(1);
    }

    warn_unused_params(params);
    return params;
  }
}
