/**
 * @file CLIOptions.cpp
 * @brief CLI argument parsing, validation, and configuration for ppforest2.
 */
#include "cli/CLIOptions.hpp"
#include "cli/Train.hpp"
#include "cli/Predict.hpp"
#include "cli/Evaluate.hpp"
#include "cli/Benchmark.hpp"
#include "cli/Summarize.hpp"
#include "cli/VarsSpec.hpp"

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <random>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <set>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef PPFOREST2_VERSION
#define PPFOREST2_VERSION "0.0.0"
#endif

namespace ppforest2::cli {
namespace {
  /**
   * @brief Check whether a CLI option was explicitly set on the command line.
   *
   * Searches the parsed subcommand first, then the global app.
   */
  bool cli_set(const std::string& flag, CLI::App * sub, CLI::App& app) {
    if (auto *opt = sub->get_option_no_throw("--" + flag))
      if (opt->count() > 0) return true;

    if (auto *opt = app.get_option_no_throw("--" + flag))
      if (opt->count() > 0) return true;

    return false;
  }

  /**
   * @brief Parse a CLI strategy string into a JSON object.
   *
   * Converts e.g. `"pda:lambda=0.3"` to `{"name": "pda", "lambda": 0.3}`.
   * Values are auto-detected as integer, float, or string.
   */
  nlohmann::json strategy_string_to_json(const std::string& input) {
    nlohmann::json j;

    auto colon = input.find(':');
    j["name"] = (colon == std::string::npos) ? input : input.substr(0, colon);

    if (colon == std::string::npos) return j;

    std::string rest = input.substr(colon + 1);
    std::istringstream ss(rest);
    std::string token;

    while (std::getline(ss, token, ',')) {
      auto eq = token.find('=');

      if (eq == std::string::npos) {
        throw std::runtime_error(
          "Invalid parameter (expected key=value): " + token);
      }

      std::string key = token.substr(0, eq);
      std::string val = token.substr(eq + 1);

      // Apply CLI → JSON key aliases
      if (key == "vars") key = "n_vars";

      // Auto-detect value type: integer → float → string
      try {
        size_t pos = 0;
        int ival   = std::stoi(val, &pos);

        if (pos == val.size()) {
          j[key] = ival;
          continue;
        }
      } catch (...) {
      }

      try {
        size_t pos  = 0;
        double dval = std::stod(val, &pos);

        if (pos == val.size()) {
          j[key] = dval;
          continue;
        }
      } catch (...) {
      }

      j[key] = val;
    }

    return j;
  }

  /**
   * @brief Load a JSON config file and apply values to CLI options.
   *
   * Config values are only applied where the CLI did not set an explicit
   * value.  Strategy objects (pp, dr, sr) flow through as JSON directly
   * without flattening to CLI strings.
   */
  void apply_config(CLIOptions & params, CLI::App& app) {
    std::ifstream file(params.config_path);

    if (!file.is_open()) {
      fmt::print(stderr, "Error: Cannot open config file: {}\n", params.config_path);
      std::exit(1);
    }

    nlohmann::json config;
    try {
      file >> config;
    } catch (const std::exception& e) {
      fmt::print(stderr, "Error: Invalid JSON in config file: {}\n", e.what());
      std::exit(1);
    }

    if (!config.is_object()) return;

    // Find the parsed subcommand
    CLI::App *sub = nullptr;
    for (auto *s : app.get_subcommands()) {
      if (s->parsed()) {
        sub = s; break;
      }
    }

    if (!sub) return;

    // Helper: apply a scalar config value if the CLI didn't set it.
    auto apply = [&](const std::string& flag, const std::string& key, auto& field) {
      if (cli_set(flag, sub, app)) return;

      if (!config.contains(key)) return;

      using T = std::decay_t<decltype(field)>;
      field   = config[key].get<T>();
    };

    // Model params
    apply("size",        "size",        params.model.size);
    apply("lambda",      "lambda",      params.model.lambda);
    apply("seed",        "seed",        params.model.seed);
    apply("threads",     "threads",     params.model.threads);
    apply("max-retries", "max_retries", params.model.max_retries);

    // Vars (string or number in config)
    if (!cli_set("vars", sub, app) && config.contains("vars")) {
      auto& v = config["vars"];

      if (v.is_string()) params.model.vars_input = v.get<std::string>();
      else if (v.is_number_integer()) params.model.vars_input = std::to_string(v.get<int>());
      else if (v.is_number_float()) params.model.vars_input = fmt::format("{:g}", v.get<double>());
    }

    // Data path
    apply("data", "data", params.data_path);

    // Evaluate params
    apply("train-ratio", "train_ratio", params.evaluate.train_ratio);
    apply("iterations",  "iterations",  params.evaluate.iterations);

    // Strategy objects: store JSON directly if CLI didn't provide
    // explicit strategy flags or shortcut params for that strategy.
    bool cli_pp = cli_set("pp", sub, app) || cli_set("lambda", sub, app);
    bool cli_dr = cli_set("dr", sub, app) || cli_set("vars", sub, app);
    bool cli_sr = cli_set("sr", sub, app);

    if (!cli_pp && config.contains("pp")) {
      if (config["pp"].is_object()) params.model.pp_config = config["pp"];
      else if (config["pp"].is_string()) params.model.pp_input = config["pp"].get<std::string>();
    }

    if (!cli_dr && config.contains("dr")) {
      if (config["dr"].is_object()) params.model.dr_config = config["dr"];
      else if (config["dr"].is_string()) params.model.dr_input = config["dr"].get<std::string>();
    }

    if (!cli_sr && config.contains("sr")) {
      if (config["sr"].is_object()) params.model.sr_config = config["sr"];
      else if (config["sr"].is_string()) params.model.sr_input = config["sr"].get<std::string>();
    }

    // Warn about unknown keys
    io::Output out(params.quiet);
    static const std::set<std::string> known = {
      "size", "lambda", "seed", "threads", "max_retries", "vars",
      "pp", "dr", "sr", "data", "train_ratio", "iterations",
    };

    for (auto it = config.begin(); it != config.end(); ++it) {
      if (known.find(it.key()) == known.end()) {
        out.println("Warning: Unknown config key '{}' — ignoring", it.key());
      }
    }
  }

  void post_parse(CLIOptions & params, CLI::App& app) {
    auto *train_sub     = app.get_subcommand("train");
    auto *predict_sub   = app.get_subcommand("predict");
    auto *eval_sub      = app.get_subcommand("evaluate");
    auto *bench_sub     = app.get_subcommand("benchmark");
    auto *summarize_sub = app.get_subcommand("summarize");

    // Determine subcommand
    if (train_sub->parsed()) {
      params.subcommand = Subcommand::train;
    } else if (predict_sub->parsed()) {
      params.subcommand = Subcommand::predict;
    } else if (eval_sub->parsed()) {
      params.subcommand = Subcommand::evaluate;
    } else if (bench_sub->parsed()) {
      params.subcommand = Subcommand::benchmark;
    } else if (summarize_sub->parsed()) {
      params.subcommand = Subcommand::summarize;
    }

    // Handle --no-save for train
    if (params.subcommand == Subcommand::train && params.no_save) {
      params.save_path.clear();
    }

    // Evaluate has no --save
    if (params.subcommand == Subcommand::evaluate) {
      params.save_path.clear();
    }

    // Apply config file (before interpreting strategy inputs)
    if (!params.config_path.empty()) {
      apply_config(params, app);
    }

    // Populate *_config from *_input strings so downstream code
    // only needs to check one field per strategy.
    auto parse_strategy_input = [](const std::string& input, const std::string& flag) {
      try {
        return strategy_string_to_json(input);
      } catch (const std::exception& e) {
        fmt::print(stderr, "Error: Invalid --{} strategy: {}\n", flag, e.what());
        std::exit(1);
      }
    };

    if (!params.model.pp_input.empty()) {
      params.model.pp_config = parse_strategy_input(params.model.pp_input, "pp");
    }

    if (!params.model.dr_input.empty()) {
      params.model.dr_config = parse_strategy_input(params.model.dr_input, "dr");
    }

    if (!params.model.sr_input.empty()) {
      params.model.sr_config = parse_strategy_input(params.model.sr_input, "sr");
    }

    // Validate strategy configs (from CLI strings or config file objects)
    if (!params.model.pp_config.is_null()) {
      try {
        pp::PPStrategy::from_json(params.model.pp_config);
      } catch (const std::exception& e) {
        fmt::print(stderr, "Error: Invalid pp strategy: {}\n", e.what());
        std::exit(1);
      }
    }

    if (!params.model.dr_config.is_null()) {
      try {
        auto dr_json = params.model.dr_config;

        if (dr_json.contains("n_vars")) {
          // n_vars can be an integer (count) or float (proportion).
          // Pass through vars_input for resolution in init_params.
          if (dr_json["n_vars"].is_number_integer()) {
            params.model.vars_input = std::to_string(dr_json["n_vars"].get<int>());
          } else {
            params.model.vars_input = fmt::format("{:g}", dr_json["n_vars"].get<double>());
          }

          // Validate with from_json. For proportions (float), substitute
          // a dummy integer so from_json can validate the strategy name.
          if (!dr_json["n_vars"].is_number_integer()) {
            dr_json["n_vars"] = 1;
          }

          dr::DRStrategy::from_json(dr_json);
        } else {
          // No n_vars (e.g. noop): validate and set sentinel
          dr::DRStrategy::from_json(dr_json);
          params.model.n_vars = 0;
          params.model.p_vars = -1;
        }
      } catch (const std::exception& e) {
        fmt::print(stderr, "Error: Invalid dr strategy: {}\n", e.what());
        std::exit(1);
      }
    }

    if (!params.model.sr_config.is_null()) {
      try {
        sr::SRStrategy::from_json(params.model.sr_config);
      } catch (const std::exception& e) {
        fmt::print(stderr, "Error: Invalid sr strategy: {}\n", e.what());
        std::exit(1);
      }
    }

    // Validate scalar ranges (these may have come from config file,
    // bypassing CLI11's range checks)
    if (params.model.lambda != -1 && (params.model.lambda < 0 || params.model.lambda > 1)) {
      fmt::print(stderr, "Error: lambda must be between 0 and 1\n");
      std::exit(1);
    }

    if (params.model.size < 0) {
      fmt::print(stderr, "Error: size must be non-negative\n");
      std::exit(1);
    }

    if (params.model.threads != -1 && params.model.threads <= 0) {
      fmt::print(stderr, "Error: threads must be positive\n");
      std::exit(1);
    }

    if (params.model.max_retries < 0) {
      fmt::print(stderr, "Error: max_retries must be non-negative\n");
      std::exit(1);
    }

    // Interpret --vars input
    if (!params.model.vars_input.empty()) {
      try {
        auto spec = parse_vars(params.model.vars_input);

        if (spec.is_proportion) {
          params.model.p_vars = spec.value;
        } else {
          params.model.n_vars = static_cast<int>(spec.value);
        }
      } catch (const std::exception& e) {
        fmt::print(stderr, "Error: Invalid --vars value: {}\n", e.what());
        std::exit(1);
      }
    }

    // Validate simulate format
    if (!params.simulation.format.empty()) {
      std::string sim_str = params.simulation.format;
      size_t x1           = sim_str.find('x');
      size_t x2           = sim_str.find('x', x1 + 1);

      if (x1 == std::string::npos || x2 == std::string::npos) {
        fmt::print(stderr, "Error: Simulate format must be NxMxK (e.g., 1000x10x2)\n");
        std::exit(1);
      }

      try {
        params.simulation.rows     = std::stoi(sim_str.substr(0, x1));
        params.simulation.cols     = std::stoi(sim_str.substr(x1 + 1, x2 - x1 - 1));
        params.simulation.n_groups = std::stoi(sim_str.substr(x2 + 1));

        if (params.simulation.rows <= 0 || params.simulation.cols <= 0 || params.simulation.n_groups <= 1) {
          throw std::out_of_range("Values must be positive and groups must be > 1");
        }
      } catch (const std::exception& e) {
        fmt::print(stderr, "Error: Invalid simulate values: {}\n", e.what());
        std::exit(1);
      }
    }

    // Evaluate requires data source
    if (params.subcommand == Subcommand::evaluate && params.simulation.format.empty() && params.data_path.empty()) {
      fmt::print(stderr, "Error: Must specify either --simulate or --data\n");
      std::exit(1);
    }

    // -i disables convergence; without -i, convergence is on
    if (params.subcommand == Subcommand::evaluate) {
      if (eval_sub->get_option("--iterations")->count() > 0) {
        params.convergence.enabled = false;
      }
    }

    io::Output out(params.quiet);
    warn_unused_params(out, params);
  }
} // anonymous namespace

  void warn_unused_params(io::Output& out, const CLIOptions& params) {
    if (params.model.size == 0) {
      bool has_warnings = false;

      if (params.model.threads != -1) {
        out.println("Warning: threads parameter is ignored when training a single tree");
        has_warnings = true;
      }

      if (params.model.p_vars != -1 || params.model.n_vars != -1) {
        out.println("Warning: --vars parameter is ignored when training a single tree");
        has_warnings = true;
      }

      if (has_warnings) {
        out.println("Single trees always use all features for splitting");
      }
    }
  }

  void init_params(CLIOptions& params, int total_vars) {
    if (params.model.lambda == -1) {
      params.model.lambda = 0.5;
    }

    if (params.evaluate.train_ratio <= 0 || params.evaluate.train_ratio >= 1) {
      fmt::print(stderr, "Error: Train ratio must be between 0 and 1\n");
      exit(1);
    }

    if (params.model.seed == -1) {
      std::random_device rd;
      params.model.seed              = rd();
      params.model.used_default_seed = true;
    }

    if (params.model.threads == -1) {
      #ifdef _OPENMP
      params.model.threads = omp_get_max_threads();
      #else
      params.model.threads = 1;
      #endif
      params.model.used_default_threads = true;
    }

    if (total_vars > 0 && params.model.size > 0) {
      if (params.model.n_vars != -1) {
        params.model.p_vars = static_cast<float>(params.model.n_vars) / total_vars;
      } else if (params.model.p_vars != -1) {
        params.model.n_vars = std::round(total_vars * params.model.p_vars);
      } else {
        params.model.p_vars            = 0.5;
        params.model.n_vars            = std::round(total_vars * params.model.p_vars);
        params.model.used_default_vars = true;
      }
    }

    // Populate strategy config defaults (now that lambda/n_vars are resolved)
    if (params.model.pp_config.is_null()) {
      params.model.pp_config = { { "name", "pda" }, { "lambda", params.model.lambda } };
    }

    if (params.model.dr_config.is_null()) {
      if (params.model.size > 0 && params.model.n_vars > 0) {
        params.model.dr_config = { { "name", "uniform" }, { "n_vars", params.model.n_vars } };
      } else {
        params.model.dr_config = { { "name", "noop" } };
      }
    }

    if (params.model.sr_config.is_null()) {
      params.model.sr_config = { { "name", "mean_of_means" } };
    }
  }

  CLIOptions parse_args(int argc, char *argv[]) {
    CLIOptions params;

    CLI::App app{ "ppforest2 - Projection Pursuit Trees and Forests" };
    app.require_subcommand(1);
    app.fallthrough();
    app.set_version_flag("--version,-V", PPFOREST2_VERSION, "Print version and exit");

    // Global options
    app.add_flag("--quiet,-q", params.quiet, "Suppress all terminal output");
    app.add_flag("--no-color", params.no_color, "Disable colored output");
    app.add_option("--config", params.config_path, "Read parameters from JSON config file")
    ->check(CLI::ExistingFile);

    // Subcommands
    setup_train(app, params);
    setup_predict(app, params);
    setup_evaluate(app, params);
    setup_benchmark(app, params);
    setup_summarize(app, params);

    // Parse
    try {
      app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
      std::exit(app.exit(e));
    }

    post_parse(params, app);
    return params;
  }
}
