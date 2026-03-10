/**
 * @file Train.cpp
 * @brief Model training utilities and train subcommand handler.
 */
#include "cli/Train.hpp"
#include "pptree.hpp"

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

using namespace pptree;
using namespace pptree::types;
using namespace pptree::stats;
using namespace pptree::io;
using json = nlohmann::json;

using pptree::variable_importance_permuted;
using pptree::variable_importance_projections;
using pptree::variable_importance_weighted_projections;

namespace pptree::cli {
namespace {
  std::string default_tag(bool is_default) {
    if (!is_default) return "";

    return " " + muted("(default)");
  }

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

    Output out(params.quiet);
    out.saved("model", path);
  }
}

  void print_configuration(
    const CLIOptions& params,
    int               n_train,
    int               n_test) {
    if (params.quiet) return;

    using namespace pptree::io;

    std::string model_type = params.trees > 0 ? "random forest" : "single decision tree";
    fmt::print("{}\n\n", emphasis("Training " + model_type));

    std::vector<Column> columns = {
      { "Parameter", 18, Align::left },
      { "Value",     30, Align::left },
    };

    Row header = header_labels(columns);
    fmt::print("  {}\n", format_row(columns, header));
    fmt::print("  {}\n", muted(format_separator(columns)));

    if (params.trees > 0) {
      fmt::print("  {}\n", format_row(columns, { "trees", std::to_string(params.trees) }));
      fmt::print("  {}\n", format_row(columns, { "variables/split",
                                                 fmt::format("{} ({}%){}", params.n_vars, static_cast<int>(params.p_vars * 100), default_tag(params.used_default_vars)) }));
      fmt::print("  {}\n", format_row(columns, { "threads",
                                                 fmt::format("{}{}", params.threads, default_tag(params.used_default_threads)) }));
      fmt::print("  {}\n", format_row(columns, { "seed",
                                                 fmt::format("{}{}", params.seed, default_tag(params.used_default_seed)) }));
    }

    std::string method = params.lambda == 0 ? "LDA" : "PDA";
    fmt::print("  {}\n", format_row(columns, { "method",
                                               fmt::format("{} (lambda={})", method, params.lambda) }));

    if (n_train > 0 && n_test > 0) {
      fmt::print("  {}\n", format_row(columns, { "training samples",
                                                 fmt::format("{} ({}%)", n_train, static_cast<int>(params.train_ratio * 100)) }));
      fmt::print("  {}\n", format_row(columns, { "test samples",
                                                 fmt::format("{} ({}%)", n_test, static_cast<int>((1 - params.train_ratio) * 100)) }));
    }

    fmt::print("\n");
  }

  DataPacket read_data(const CLIOptions& params, pptree::stats::RNG& rng) {
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

  TrainResult train_model(
    const FeatureMatrix&  x,
    const ResponseVector& y,
    const CLIOptions&     params,
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

  int run_train(CLIOptions& params) {
    Output out(params.quiet);

    // Validate save path before training
    if (!params.save_path.empty()) {
      check_file_not_exists(params.save_path);
    }

    pptree::stats::RNG rng(params.seed);
    auto data = read_data(params, rng);

    init_params(params, data.x.cols());
    print_configuration(params);

    FeatureMatrix x  = data.x;
    ResponseVector y = data.y;

    const auto train_result = train_model(x, y, params, rng);

    out.print("  Trained in {}ms, ", emphasis(std::to_string(train_result.duration)));

    if (!params.save_path.empty()) {
      save_model(*train_result.model, params, params.save_path);
    } else {
      out.print("not saved {}\n", muted("(used --no-save)"));
    }

    if (!params.no_metrics) {
      const auto *forest = dynamic_cast<const Forest *>(train_result.model.get());
      const auto *tree   = dynamic_cast<const Tree *>(train_result.model.get());
      VariableImportance vi;
      double oob_err = -1.0;

      const int n_vars = static_cast<int>(x.cols());

      vi.scale = stats::sd(x);
      vi.scale = (vi.scale.array() > Feature(0)).select(vi.scale, Feature(1));

      if (forest != nullptr) {
        oob_err                 = forest->oob_error(x, y);
        vi.permuted             = variable_importance_permuted(*forest, x, y, params.seed);
        vi.projections          = variable_importance_projections(*forest, n_vars, &vi.scale);
        vi.weighted_projections = variable_importance_weighted_projections(*forest, x, y, &vi.scale);
      } else if (tree != nullptr) {
        vi.projections = variable_importance_projections(*tree, n_vars, &vi.scale);
      }

      if (!out.quiet) {
        if (oob_err >= 0.0) {
          out.print("  OOB error: {}\n\n", emphasis(fmt::format("{:.2f}%", oob_err * 100)));
        }

        print_variable_importance(vi);
      }

      if (!params.save_path.empty()) {
        std::ifstream in(params.save_path);
        json saved = json::parse(in);
        in.close();

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
