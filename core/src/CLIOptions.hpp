#pragma once

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pptree {
  class ConfigJSON : public CLI::Config {
    public:
      ConfigJSON(std::vector<std::string> subcommand_names = {})
        : subcommand_names_(std::move(subcommand_names)) {}

      std::string to_config(const CLI::App *app, bool default_also, bool, std::string) const override {
        nlohmann::json j;

        for (const CLI::Option *opt : app->get_options({})) {
          if (!opt->get_lnames().empty() && opt->get_configurable()) {
            std::string name = opt->get_lnames()[0];

            if (opt->get_type_size() != 0) {
              if (opt->count() == 1)
                j[name] = opt->results().at(0);
              else if (opt->count() > 1)
                j[name] = opt->results();
              else if (default_also && !opt->get_default_str().empty())
                j[name] = opt->get_default_str();
            } else if (opt->count() == 1) {
              j[name] = true;
            } else if (opt->count() > 1) {
              j[name] = opt->count();
            } else if (opt->count() == 0 && default_also) {
              j[name] = false;
            }
          }
        }

        for (const CLI::App *subcom : app->get_subcommands({}))
          j[subcom->get_name()] = nlohmann::json(to_config(subcom, default_also, false, ""));

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

            if (!name.empty())
              copy_prefix.push_back(name);

            auto sub_results = _from_config(*item, item.key(), copy_prefix);
            results.insert(results.end(), sub_results.begin(), sub_results.end());
          }
        } else if (!name.empty()) {
          if (prefix.empty() && !subcommand_names_.empty()) {
            for (const auto& sub_name : subcommand_names_) {
              results.emplace_back();
              CLI::ConfigItem &res = results.back();
              res.name    = name;
              res.parents = {sub_name};
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
          res.inputs = {j.get<bool>() ? "true" : "false"};
        } else if (j.is_number()) {
          std::stringstream ss;
          ss << j.get<double>();
          res.inputs = {ss.str()};
        } else if (j.is_string()) {
          res.inputs = {j.get<std::string>()};
        } else if (j.is_array()) {
          for (const auto &val : j)
            res.inputs.push_back(val.get<std::string>());
        } else {
          throw CLI::ConversionError("Failed to convert " + res.name);
        }
      }
  };
  enum class OutputFormat { text, json };
  enum class Subcommand { none, train, predict, evaluate };

  struct CLIOptions {
    int trees         = 100;
    float lambda      = 0.5;
    int threads       = -1;
    int seed          = -1;
    float p_vars      = 0.5;
    int n_vars        = -1;
    float train_ratio = 0.7;
    int n_runs        = 1;
    std::string data_path;
    std::string simulate;
    int rows                  = 1000;
    int cols                  = 10;
    int classes               = 2;
    float sim_mean            = 100.0f;
    float sim_mean_separation = 50.0f;
    float sim_sd              = 10.0f;

    Subcommand subcommand       = Subcommand::none;
    std::string save_path;
    std::string model_path;
    OutputFormat output_format  = OutputFormat::text;
    bool quiet                  = false;
  };

  void warn_unused_params(const CLIOptions& params) {
    if (params.quiet) return;

    if (params.trees == 0) {
      bool has_warnings = false;

      if (params.threads != -1) {
        std::cout << "Warning: threads parameter is ignored when training a single tree" << std::endl;
        has_warnings = true;
      }

      if (params.p_vars != 0.5) {
        std::cout << "Warning: var-proportion parameter is ignored when training a single tree" << std::endl;
        has_warnings = true;
      }

      if (has_warnings) {
        std::cout << "Single trees always use all features for splitting" << std::endl;
      }
    }
  }

  void init_params(CLIOptions& params, int total_vars = 0) {
    if (params.lambda == -1) {
      params.lambda = 0.5;

      if (!params.quiet) {
        std::cout << "Using default lambda: " << params.lambda << std::endl;
      }
    }

    if (params.train_ratio <= 0 || params.train_ratio >= 1) {
      std::cerr << "Error: Train ratio must be between 0 and 1" << std::endl;
      exit(1);
    }

    if (params.seed == -1) {
      std::random_device rd;
      params.seed = rd();

      if (!params.quiet) {
        std::cout << "Using random seed: " << params.seed << std::endl;
      }
    }

    if (params.threads == -1) {
     #ifdef _OPENMP
      params.threads = omp_get_max_threads();
     #else
      params.threads = 1;
     #endif

      if (!params.quiet) {
        std::cout << "Using default thread count: " << params.threads << std::endl;
      }
    }

    if (total_vars > 0 && params.trees > 0) {
      if (params.p_vars == -1 && params.n_vars == -1) {
        params.p_vars = 0.5;
        params.n_vars = std::round(total_vars * params.p_vars);

        if (!params.quiet) {
          std::cout << "Using default variable proportion: " << params.p_vars  << " (" << params.n_vars << " variables)" << std::endl;
        }
      } else if (params.p_vars != -1) {
        params.n_vars = std::round(total_vars * params.p_vars);
      } else {
        params.p_vars = static_cast<float>(params.n_vars) / total_vars;
      }
    }
  }

  CLIOptions parse_args(int argc, char *argv[]) {
    CLIOptions params;

    CLI::App app{"pptree - Projection Pursuit Trees and Forests"};
    app.require_subcommand(1);

    // Global options
    std::string output_format_str = "text";
    app.add_option("--output-format", output_format_str, "Output format: text or json")
       ->check(CLI::IsMember({"text", "json"}));
    app.add_flag("--quiet,-q", params.quiet, "Suppress progress bars and informational output");
    app.config_formatter(std::make_shared<ConfigJSON>(
      std::vector<std::string>{"train", "predict", "evaluate"}));
    app.set_config("--config", "", "Read parameters from JSON config file");

    // Helper: add model training options shared by train and evaluate
    auto add_model_options = [&](CLI::App* sub) {
      sub->add_option("-t,--trees", params.trees, "Number of trees (default: 100, 0 for single tree)")
         ->check(CLI::NonNegativeNumber);
      sub->add_option("-l,--lambda", params.lambda, "Method selection (0=LDA, (0,1]=PDA)")
         ->check(CLI::Range(0.0f, 1.0f));
      sub->add_option("-n,--threads", params.threads, "Number of threads (default: CPU cores)")
         ->check(CLI::PositiveNumber);
      sub->add_option("-r,--seed", params.seed, "Random seed (default: random)");
      sub->add_option("-v,--p-vars", params.p_vars, "Feature proportion for forest")
         ->check(CLI::Range(0.0f, 1.0f));
      sub->add_option("-m,--n-vars", params.n_vars, "Number of features per split")
         ->check(CLI::PositiveNumber);
    };

    // Train subcommand
    auto train_sub = app.add_subcommand("train", "Train a model");
    train_sub->add_option("-d,--data", params.data_path, "CSV training data")
             ->required()
             ->check(CLI::ExistingFile);
    add_model_options(train_sub);
    train_sub->add_option("-o,--save", params.save_path, "Save trained model to JSON file");

    // Predict subcommand
    auto predict_sub = app.add_subcommand("predict", "Load a model and predict on new data");
    predict_sub->add_option("-M,--model", params.model_path, "Saved model JSON file")
               ->required()
               ->check(CLI::ExistingFile);
    predict_sub->add_option("-d,--data", params.data_path, "CSV data to predict on")
               ->required()
               ->check(CLI::ExistingFile);

    // Evaluate subcommand
    auto eval_sub = app.add_subcommand("evaluate", "Train and evaluate a model");
    auto eval_data_opt = eval_sub->add_option("-d,--data", params.data_path, "CSV file")
                                 ->check(CLI::ExistingFile);
    auto eval_sim_opt  = eval_sub->add_option("-s,--simulate", params.simulate, "Simulate NxMxK data");
    eval_data_opt->excludes(eval_sim_opt);
    eval_sim_opt->excludes(eval_data_opt);

    add_model_options(eval_sub);
    eval_sub->add_option("-p,--train-ratio", params.train_ratio, "Train set ratio (default: 0.7)")
            ->check(CLI::Range(0.01f, 0.99f));
    eval_sub->add_option("-e,--n-runs", params.n_runs, "Number of training runs (default: 1)")
            ->check(CLI::PositiveNumber);
    eval_sub->add_option("--sim-mean", params.sim_mean, "Mean for simulated data (default: 100.0)")
            ->needs(eval_sim_opt);
    eval_sub->add_option("--sim-mean-separation", params.sim_mean_separation, "Mean separation between classes (default: 50.0)")
            ->needs(eval_sim_opt)
            ->check(CLI::PositiveNumber);
    eval_sub->add_option("--sim-sd", params.sim_sd, "Standard deviation for simulated data (default: 10.0)")
            ->needs(eval_sim_opt)
            ->check(CLI::PositiveNumber);

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

    // Post-parse: map output format
    if (output_format_str == "json") {
      params.output_format = OutputFormat::json;
      params.quiet = true;
    }

    // Post-parse: validate simulate format
    if (!params.simulate.empty()) {
      std::string sim_str = params.simulate;
      size_t x1 = sim_str.find('x');
      size_t x2 = sim_str.find('x', x1 + 1);

      if (x1 == std::string::npos || x2 == std::string::npos) {
        std::cerr << "Error: Simulate format must be NxMxK (e.g., 1000x10x2)" << std::endl;
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
        std::cerr << "Error: Invalid simulate values: " << e.what() << std::endl;
        std::exit(1);
      }
    }

    // Post-parse: evaluate requires data source
    if (params.subcommand == Subcommand::evaluate && params.simulate.empty() && params.data_path.empty()) {
      std::cerr << "Error: Must specify either --simulate or --data" << std::endl;
      std::exit(1);
    }

    warn_unused_params(params);
    return params;
  }
}
