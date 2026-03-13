/**
 * @file Train.cpp
 * @brief Model training utilities and train subcommand handler.
 */
#include "cli/Train.hpp"
#include "ppforest2.hpp"
#include "models/ModelVisitor.hpp"

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <fstream>

#include "stats/Simulation.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "io/Presentation.hpp"
#include "io/Color.hpp"
#include "io/Output.hpp"
#include "io/Table.hpp"
#include "io/IO.hpp"
#include "io/Timing.hpp"
#include "serialization/Json.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using namespace ppforest2::io;
using json = nlohmann::json;

using ppforest2::variable_importance_permuted;
using ppforest2::variable_importance_projections;
using ppforest2::variable_importance_weighted_projections;

namespace ppforest2::cli {
  void add_model_options(CLI::App *sub, ModelParams& model) {
    sub->add_option("-t,--trees", model.trees, "Number of trees (default: 100, 0 for single tree)")
    ->check(CLI::NonNegativeNumber);
    sub->add_option("-l,--lambda", model.lambda, "Method selection (0=LDA, (0,1]=PDA)")
    ->check(CLI::Range(0.0f, 1.0f));
    sub->add_option("--threads", model.threads, "Number of threads (default: CPU cores)")
    ->check(CLI::PositiveNumber);
    sub->add_option("-r,--seed", model.seed, "Random seed (default: random)");
    sub->add_option("-v,--vars", model.vars_input, "Features per split (integer=count, decimal or fraction=proportion, default: 0.5)");
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
namespace {
  std::string default_tag(bool is_default) {
    if (!is_default) return "";

    return " " + muted("(default)");
  }

  void save_model(
    Output &            out,
    const Model&       model,
    const CLIOptions&  params,
    const std::string& path) {
    json output = serialization::to_json(model);

    json config;
    config["trees"]   = params.model.trees;
    config["lambda"]  = params.model.lambda;
    config["seed"]    = params.model.seed;
    config["threads"] = params.model.threads;

    if (params.model.trees > 0 && params.model.n_vars > 0) {
      config["vars"] = params.model.n_vars;
    }

    if (!params.data_path.empty()) {
      config["data"] = params.data_path;
    }

    output["config"] = config;

    write_json_file(output, path);

    out.saved("model", path);
  }
}

  void print_configuration(
    Output&           out,
    const CLIOptions& params,
    int               n_train,
    int               n_test) {
    std::string model_type = params.model.trees > 0 ? "random forest" : "single decision tree";
    out.println("{}", emphasis("Training " + model_type));
    out.newline();

    std::vector<Column> columns = {
      { "Parameter", 18, Align::left },
      { "Value",     30, Align::left },
    };

    out.indent();

    Row header = header_labels(columns);
    out.println("{}", format_row(columns, header));
    out.println("{}", muted(format_separator(columns)));

    if (params.model.trees > 0) {
      out.println("{}", format_row(columns, { "trees", std::to_string(params.model.trees) }));
      out.println("{}", format_row(columns, { "variables/split",
                                              fmt::format("{} ({}%){}", params.model.n_vars, static_cast<int>(params.model.p_vars * 100), default_tag(params.model.used_default_vars)) }));
      out.println("{}", format_row(columns, { "threads",
                                              fmt::format("{}{}", params.model.threads, default_tag(params.model.used_default_threads)) }));
      out.println("{}", format_row(columns, { "seed",
                                              fmt::format("{}{}", params.model.seed, default_tag(params.model.used_default_seed)) }));
    }

    std::string method = params.model.lambda == 0 ? "LDA" : "PDA";
    out.println("{}", format_row(columns, { "method",
                                            fmt::format("{} (lambda={})", method, params.model.lambda) }));

    if (n_train > 0 && n_test > 0) {
      out.println("{}", format_row(columns, { "training samples",
                                              fmt::format("{} ({}%)", n_train, static_cast<int>(params.evaluate.train_ratio * 100)) }));
      out.println("{}", format_row(columns, { "test samples",
                                              fmt::format("{} ({}%)", n_test, static_cast<int>((1 - params.evaluate.train_ratio) * 100)) }));
    }

    out.dedent();
    out.newline();
  }

  DataPacket read_data(const CLIOptions& params, ppforest2::stats::RNG& rng) {
    if (!params.data_path.empty()) {
      try {
        return read_csv_sorted(params.data_path);
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
        return simulate(params.simulation.rows, params.simulation.cols, params.simulation.classes, rng, simulation_params);
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
    TrainingSpec::Ptr spec;
    std::function<Model::Ptr()> fact;

    if (params.model.trees > 0) {
      spec = TrainingSpecUGLDA::make(params.model.n_vars, params.model.lambda);
      fact = [&] {
        return std::make_unique<Forest>(Forest::train(*spec, x, y, params.model.trees, params.model.seed, params.model.threads));
      };
    } else {
      spec = TrainingSpecGLDA::make(params.model.lambda);
      fact = [&] {
        return std::make_unique<Tree>(Tree::train(*spec, x, y, rng));
      };
    }

    auto [model, ms] = measure_time_ms(fact);

    return { std::move(model), ms };
  }

  int run_train(CLIOptions& params) {
    Output out(params.quiet);

    // Validate save path before training
    if (!params.save_path.empty()) {
      check_file_not_exists(params.save_path);
    }

    ppforest2::stats::RNG rng(params.model.seed);
    auto data = read_data(params, rng);

    init_params(params, data.x.cols());
    print_configuration(out, params);

    FeatureMatrix x  = data.x;
    ResponseVector y = data.y;

    const auto train_result = train_model(x, y, params, rng);

    out.indent();
    out.println("Trained in {}ms", emphasis(std::to_string(train_result.duration)));

    if (!params.save_path.empty()) {
      save_model(out, *train_result.model, params, params.save_path);
    } else {
      out.println("not saved {}", muted("(used --no-save)"));
    }

    out.dedent();

    if (!params.no_metrics) {
      VariableImportance vi;
      double oob_err = -1.0;

      const int n_vars = static_cast<int>(x.cols());

      vi.scale = stats::sd(x);
      vi.scale = (vi.scale.array() > Feature(0)).select(vi.scale, Feature(1));

      struct MetricsVisitor : ModelVisitor {
        const FeatureMatrix& x;
        const ResponseVector& y;
        const CLIOptions& params;
        int n_vars;
        VariableImportance& vi;
        double& oob_err;

        MetricsVisitor(const FeatureMatrix& x, const ResponseVector& y,
          const CLIOptions& params, int n_vars,
          VariableImportance& vi, double& oob_err) :
          x(x), y(y), params(params), n_vars(n_vars), vi(vi), oob_err(oob_err) {}

        void visit(const Forest& forest) override {
          oob_err                 = forest.oob_error(x, y);
          vi.permuted             = variable_importance_permuted(forest, x, y, params.model.seed);
          vi.projections          = variable_importance_projections(forest, n_vars, &vi.scale);
          vi.weighted_projections = variable_importance_weighted_projections(forest, x, y, &vi.scale);
        }

        void visit(const Tree& tree) override {
          vi.projections = variable_importance_projections(tree, n_vars, &vi.scale);
        }
      };

      MetricsVisitor metrics_visitor(x, y, params, n_vars, vi, oob_err);
      train_result.model->accept(metrics_visitor);

      if (oob_err >= 0.0) {
        out.indent();
        out.println("OOB error: {}", emphasis(fmt::format("{:.2f}%", oob_err * 100)));
        out.dedent();
        out.newline();
      }

      print_variable_importance(out, vi);

      if (!params.save_path.empty()) {
        json saved = read_json_file(params.save_path);

        if (oob_err >= 0.0) {
          saved["oob_error"] = oob_err;
        }

        saved["variable_importance"] = serialization::to_json(vi);
        write_json_file(saved, params.save_path);
      }
    }

    return 0;
  }
}
