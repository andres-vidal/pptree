/**
 * @file Summarize.cpp
 * @brief Summarize subcommand handler: display saved model summary.
 */
#include "cli/Summarize.hpp"
#include "cli/Metrics.hpp"
#include "io/Presentation.hpp"
#include "io/Output.hpp"
#include "io/IO.hpp"
#include "serialization/Json.hpp"
#include "utils/UserError.hpp"

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ppforest2::cli {
  CLI::App * setup_summarize(CLI::App& app, CLIOptions& params) {
    auto sub = app.add_subcommand("summarize", "Display a saved model summary");
    sub->add_option("-M,--model", params.model_path, "Saved model JSON file")
    ->required()
    ->check(CLI::ExistingFile);
    sub->add_option("-d,--data", params.data_path, "CSV training data (recomputes metrics if provided)")
    ->check(CLI::ExistingFile);
    return sub;
  }

  int run_summarize(CLIOptions& params) {
    io::Output out(params.quiet);

    json model_data = io::json::read_file(params.model_path);

    // Recompute metrics if data is provided and metrics are absent
    bool has_metrics = model_data.contains("training_confusion_matrix") || model_data.contains("variable_importance");

    if (!params.data_path.empty() && !has_metrics) {
      try {
        auto data = io::csv::read_sorted(params.data_path);

        std::vector<std::string> group_names;

        if (model_data.contains("meta") && model_data["meta"].contains("groups")) {
          group_names = model_data["meta"]["groups"].get<std::vector<std::string>>();
        }

        auto model = serialization::model_from_json(model_data);
        compute_metrics(model_data, *model, data.x, data.y, group_names);
      } catch (const ppforest2::UserError& e) {
        fmt::print(stderr, "Error: {}\n", e.what());
        fmt::print(stderr, "File: {}\n", params.data_path);
        return 1;
      } catch (const std::exception& e) {
        fmt::print(stderr, "Error reading data: {}\n", e.what());
        return 1;
      }
    }

    io::print_summary(out, model_data);

    return 0;
  }
}
