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
#include "ppforest2.hpp"

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

namespace ppforest2::cli {
  nlohmann::json strategy_string_to_json(std::string const& input) {
    nlohmann::json j;

    auto colon = input.find(':');
    j["name"]  = (colon == std::string::npos) ? input : input.substr(0, colon);

    if (colon == std::string::npos) {
      return j;
    }

    std::string rest = input.substr(colon + 1);
    std::istringstream ss(rest);
    std::string token;

    while (std::getline(ss, token, ',')) {
      auto eq = token.find('=');

      if (eq == std::string::npos) {
        throw std::runtime_error("Invalid parameter (expected key=value): " + token);
      }

      std::string key = token.substr(0, eq);
      std::string val = token.substr(eq + 1);

      // Auto-detect value type: integer → float → string
      try {
        size_t pos = 0;
        int ival   = std::stoi(val, &pos);

        if (pos == val.size()) {
          j[key] = ival;
          continue;
        }
      } catch (...) {}

      try {
        size_t pos  = 0;
        double dval = std::stod(val, &pos);

        if (pos == val.size()) {
          j[key] = dval;
          continue;
        }
      } catch (...) {}

      j[key] = val;
    }

    return j;
  }

  namespace {
    /**
   * @brief Check whether a CLI option was explicitly set on the command line.
   *
   * Searches the parsed subcommand first, then the global app.
   */
    bool cli_set(std::string const& flag, CLI::App* sub, CLI::App& app) {
      if (auto* opt = sub->get_option_no_throw("--" + flag))
        if (opt->count() > 0)
          return true;

      if (auto* opt = app.get_option_no_throw("--" + flag))
        if (opt->count() > 0)
          return true;

      return false;
    }

    /**
   * @brief Load a JSON config file and apply values to CLI options.
   *
   * Config values are only applied where the CLI did not set an explicit
   * value.  Strategy objects (pp, vars, cutpoint) flow through as JSON directly
   * without flattening to CLI strings.
   */
    void apply_config(CLIOptions& params, CLI::App& app) {
      std::ifstream file(params.config_path);

      if (!file.is_open()) {
        fmt::print(stderr, "Error: Cannot open config file: {}\n", params.config_path);
        std::exit(1);
      }

      nlohmann::json config;
      try {
        file >> config;
      } catch (std::exception const& e) {
        fmt::print(stderr, "Error: Invalid JSON in config file: {}\n", e.what());
        std::exit(1);
      }

      if (!config.is_object())
        return;

      // Find the parsed subcommand
      CLI::App* sub = nullptr;
      for (auto* s : app.get_subcommands()) {
        if (s->parsed()) {
          sub = s;
          break;
        }
      }

      if (!sub)
        return;

      // Helper: apply a scalar config value if the CLI didn't set it.
      auto apply = [&](std::string const& flag, std::string const& key, auto& field) {
        if (cli_set(flag, sub, app))
          return;

        if (!config.contains(key))
          return;

        using T = std::decay_t<decltype(field)>;
        field   = config[key].get<T>();
      };

      // Model params
      apply("size", "size", params.model.size);
      apply("lambda", "lambda", params.model.lambda);
      apply("seed", "seed", params.model.seed);
      apply("threads", "threads", params.model.threads);
      apply("max-retries", "max_retries", params.model.max_retries);

      // Vars shortcuts (n-vars / p-vars from config)
      if (!cli_set("n-vars", sub, app) && !cli_set("p-vars", sub, app)) {
        if (config.contains("n_vars") && config["n_vars"].is_number_integer()) {
          params.model.n_vars = config["n_vars"].get<int>();
        } else if (config.contains("p_vars")) {
          auto& v = config["p_vars"];

          if (v.is_number_float() || v.is_number_integer())
            params.model.p_vars_input = fmt::format("{:g}", v.get<double>());
          else if (v.is_string())
            params.model.p_vars_input = v.get<std::string>();
        }
      }

      // Data path
      apply("data", "data", params.data_path);

      // Evaluate params
      apply("train-ratio", "train_ratio", params.evaluate.train_ratio);
      apply("iterations", "iterations", params.evaluate.iterations);

      // Strategy objects: store JSON directly if CLI didn't provide
      // explicit strategy flags or shortcut params for that strategy.
      bool cli_pp        = cli_set("pp", sub, app) || cli_set("lambda", sub, app);
      bool cli_vars      = cli_set("vars", sub, app) || cli_set("n-vars", sub, app) || cli_set("p-vars", sub, app);
      bool cli_cutpoint  = cli_set("cutpoint", sub, app);
      bool cli_stop      = cli_set("stop", sub, app);
      bool cli_binarize  = cli_set("binarize", sub, app);
      bool cli_partition = cli_set("partition", sub, app);
      bool cli_leaf      = cli_set("leaf", sub, app);

      if (!cli_pp && config.contains("pp")) {
        if (config["pp"].is_object())
          params.model.pp_config = config["pp"];
        else if (config["pp"].is_string())
          params.model.pp_input = config["pp"].get<std::string>();
      }

      if (!cli_vars && config.contains("vars")) {
        if (config["vars"].is_object())
          params.model.vars_config = config["vars"];
        else if (config["vars"].is_string())
          params.model.vars_input = config["vars"].get<std::string>();
      }

      if (!cli_cutpoint && config.contains("cutpoint")) {
        if (config["cutpoint"].is_object())
          params.model.cutpoint_config = config["cutpoint"];
        else if (config["cutpoint"].is_string())
          params.model.cutpoint_input = config["cutpoint"].get<std::string>();
      }

      if (!cli_stop && config.contains("stop")) {
        if (config["stop"].is_object())
          params.model.stop_config = config["stop"];
        else if (config["stop"].is_string())
          params.model.stop_input = config["stop"].get<std::string>();
      }

      if (!cli_binarize && config.contains("binarize")) {
        if (config["binarize"].is_object())
          params.model.binarize_config = config["binarize"];
        else if (config["binarize"].is_string())
          params.model.binarize_input = config["binarize"].get<std::string>();
      }

      if (!cli_partition && config.contains("partition")) {
        if (config["partition"].is_object())
          params.model.partition_config = config["partition"];
        else if (config["partition"].is_string())
          params.model.partition_input = config["partition"].get<std::string>();
      }

      if (!cli_leaf && config.contains("leaf")) {
        if (config["leaf"].is_object())
          params.model.leaf_config = config["leaf"];
        else if (config["leaf"].is_string())
          params.model.leaf_input = config["leaf"].get<std::string>();
      }

      // Warn about unknown keys
      io::Output out(params.quiet);
      static std::set<std::string> const known = {
          "size",
          "lambda",
          "seed",
          "threads",
          "max_retries",
          "n_vars",
          "p_vars",
          "vars",
          "pp",
          "cutpoint",
          "stop",
          "binarize",
          "partition",
          "leaf",
          "data",
          "train_ratio",
          "iterations",
      };

      for (auto it = config.begin(); it != config.end(); ++it) {
        if (known.find(it.key()) == known.end()) {
          out.println("Warning: Unknown config key '{}' — ignoring", it.key());
        }
      }
    }

    void post_parse(CLIOptions& params, CLI::App& app) {
      auto* train_sub     = app.get_subcommand("train");
      auto* predict_sub   = app.get_subcommand("predict");
      auto* eval_sub      = app.get_subcommand("evaluate");
      auto* bench_sub     = app.get_subcommand("benchmark");
      auto* summarize_sub = app.get_subcommand("summarize");

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
      auto parse_strategy_input = [](std::string const& input, std::string const& flag) {
        try {
          return strategy_string_to_json(input);
        } catch (std::exception const& e) {
          fmt::print(stderr, "Error: Invalid --{} strategy: {}\n", flag, e.what());
          std::exit(1);
        }
      };

      if (!params.model.pp_input.empty()) {
        params.model.pp_config = parse_strategy_input(params.model.pp_input, "pp");
      }

      if (!params.model.vars_input.empty()) {
        params.model.vars_config = parse_strategy_input(params.model.vars_input, "vars");
      }

      if (!params.model.cutpoint_input.empty()) {
        params.model.cutpoint_config = parse_strategy_input(params.model.cutpoint_input, "cutpoint");
      }

      if (!params.model.stop_input.empty()) {
        params.model.stop_config = parse_strategy_input(params.model.stop_input, "stop");
      }

      if (!params.model.binarize_input.empty()) {
        params.model.binarize_config = parse_strategy_input(params.model.binarize_input, "binarize");
      }

      if (!params.model.partition_input.empty()) {
        params.model.partition_config = parse_strategy_input(params.model.partition_input, "partition");
      }

      if (!params.model.leaf_input.empty()) {
        params.model.leaf_config = parse_strategy_input(params.model.leaf_input, "leaf");
      }

      // Validate strategy configs (from CLI strings or config file objects)
      if (!params.model.pp_config.is_null()) {
        try {
          pp::ProjectionPursuit::from_json(params.model.pp_config);
        } catch (std::exception const& e) {
          fmt::print(stderr, "Error: Invalid pp strategy: {}\n", e.what());
          std::exit(1);
        }
      }

      if (!params.model.vars_config.is_null()) {
        try {
          auto vars_json = params.model.vars_config;

          if (vars_json.contains("count")) {
            // count can be an integer (count) or float (proportion).
            if (vars_json["count"].is_number_integer()) {
              params.model.n_vars = vars_json["count"].get<int>();
            } else {
              params.model.p_vars_input = fmt::format("{:g}", vars_json["count"].get<double>());
            }

            // Validate with from_json. For proportions (float), substitute
            // a dummy integer so from_json can validate the strategy name.
            if (!vars_json["count"].is_number_integer()) {
              vars_json["count"] = 1;
            }

            vars::VariableSelection::from_json(vars_json);
          } else {
            // No count (e.g. all): validate and set sentinel
            vars::VariableSelection::from_json(vars_json);
            params.model.n_vars = 0;
            params.model.p_vars = -1;
          }
        } catch (std::exception const& e) {
          fmt::print(stderr, "Error: Invalid vars strategy: {}\n", e.what());
          std::exit(1);
        }
      }

      if (!params.model.cutpoint_config.is_null()) {
        try {
          cutpoint::SplitCutpoint::from_json(params.model.cutpoint_config);
        } catch (std::exception const& e) {
          fmt::print(stderr, "Error: Invalid cutpoint strategy: {}\n", e.what());
          std::exit(1);
        }
      }

      if (!params.model.stop_config.is_null()) {
        try {
          stop::StopRule::from_json(params.model.stop_config);
        } catch (std::exception const& e) {
          fmt::print(stderr, "Error: Invalid stop strategy: {}\n", e.what());
          std::exit(1);
        }
      }

      if (!params.model.binarize_config.is_null()) {
        try {
          binarize::Binarization::from_json(params.model.binarize_config);
        } catch (std::exception const& e) {
          fmt::print(stderr, "Error: Invalid binarize strategy: {}\n", e.what());
          std::exit(1);
        }
      }

      if (!params.model.partition_config.is_null()) {
        try {
          partition::StepPartition::from_json(params.model.partition_config);
        } catch (std::exception const& e) {
          fmt::print(stderr, "Error: Invalid partition strategy: {}\n", e.what());
          std::exit(1);
        }
      }

      if (!params.model.leaf_config.is_null()) {
        try {
          leaf::LeafStrategy::from_json(params.model.leaf_config);
        } catch (std::exception const& e) {
          fmt::print(stderr, "Error: Invalid leaf strategy: {}\n", e.what());
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

      // Interpret --p-vars input (fraction or decimal proportion)
      if (!params.model.p_vars_input.empty()) {
        try {
          auto spec = parse_vars(params.model.p_vars_input);

          if (spec.is_proportion) {
            params.model.p_vars = spec.value;
          } else {
            // User passed an integer to --p-vars, treat as error
            fmt::print(
                stderr, "Error: --p-vars expects a proportion (e.g. 0.5 or 1/2), use --n-vars for integer counts\n"
            );
            std::exit(1);
          }
        } catch (std::exception const& e) {
          fmt::print(stderr, "Error: Invalid --p-vars value: {}\n", e.what());
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
        } catch (std::exception const& e) {
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

  void warn_unused_params(io::Output& out, CLIOptions const& params) {
    if (params.model.size == 0) {
      bool has_warnings = false;

      if (params.model.threads != -1) {
        out.println("Warning: threads parameter is ignored when training a single tree");
        has_warnings = true;
      }

      if (params.model.p_vars != -1 || params.model.n_vars != -1) {
        out.println("Warning: --n-vars/--p-vars parameter is ignored when training a single tree");
        has_warnings = true;
      }

      if (has_warnings) {
        out.println("Single trees always use all features for splitting");
      }
    }
  }

  int proportion_to_count(float p, unsigned int total) {
    return static_cast<int>(std::round(static_cast<float>(total) * p));
  }

  void init_params(CLIOptions& params, unsigned int total_vars) {
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
        params.model.p_vars = static_cast<float>(params.model.n_vars) / static_cast<float>(total_vars);
      } else if (params.model.p_vars != -1) {
        params.model.n_vars = proportion_to_count(params.model.p_vars, total_vars);
      } else {
        params.model.p_vars            = 0.5;
        params.model.n_vars            = proportion_to_count(params.model.p_vars, total_vars);
        params.model.used_default_vars = true;
      }
    }

    // Populate strategy config defaults (now that lambda/n_vars are resolved)
    if (params.model.pp_config.is_null()) {
      params.model.pp_config = {{"name", "pda"}, {"lambda", params.model.lambda}};
    }

    if (params.model.vars_config.is_null()) {
      if (params.model.size > 0 && params.model.n_vars > 0) {
        params.model.vars_config = {{"name", "uniform"}, {"count", params.model.n_vars}};
      } else {
        params.model.vars_config = {{"name", "all"}};
      }
    }

    if (params.model.cutpoint_config.is_null()) {
      params.model.cutpoint_config = {{"name", "mean_of_means"}};
    }
  }

  CLIOptions parse_args(int argc, char* argv[]) {
    CLIOptions params;

    CLI::App app{"ppforest2 - Projection Pursuit Trees and Forests"};
    app.require_subcommand(1);
    app.fallthrough();
    app.set_version_flag("--version,-V", PPFOREST2_VERSION, "Print version and exit");

    // Global options
    app.add_flag("--quiet,-q", params.quiet, "Suppress all terminal output");
    app.add_flag("--no-color", params.no_color, "Disable colored output");
    app.add_option("--config", params.config_path, "Read parameters from JSON config file")->check(CLI::ExistingFile);

    // Subcommands
    setup_train(app, params);
    setup_predict(app, params);
    setup_evaluate(app, params);
    setup_benchmark(app, params);
    setup_summarize(app, params);

    // Parse
    try {
      app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
      std::exit(app.exit(e));
    }

    post_parse(params, app);
    return params;
  }
}
