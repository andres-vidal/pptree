/**
 * @file Predict.cpp
 * @brief Predict subcommand handler.
 */
#include "cli/Predict.hpp"
#include "ppforest2.hpp"

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <fstream>
#include <vector>

#include "stats/DataPacket.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "io/Presentation.hpp"
#include "io/Color.hpp"
#include "io/Output.hpp"
#include "io/IO.hpp"
#include "serialization/Json.hpp"
#include "utils/UserError.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using namespace ppforest2::io::style;
using json = nlohmann::json;

namespace ppforest2::cli {
  CLI::App * setup_predict(CLI::App& app, CLIOptions& params) {
    auto sub = app.add_subcommand("predict", "Load a model and predict on new data");
    sub->add_option("-M,--model", params.model_path, "Saved model JSON file")
    ->required()
    ->check(CLI::ExistingFile);
    sub->add_option("-d,--data", params.data_path, "CSV data to predict on")
    ->required()
    ->check(CLI::ExistingFile);
    sub->add_option("-o,--output", params.output_path, "Save prediction results to JSON file");
    sub->add_flag("--no-metrics", params.no_metrics, "Omit error rate and confusion matrix from output");
    return sub;
  }
}

namespace ppforest2::cli {
namespace {
  json build_predict_result(
    const ResponseVector& predictions,
    const DataPacket&     data,
    bool no_metrics) {
    json result;
    std::vector<int> pred_vec(predictions.data(), predictions.data() + predictions.size());
    result["predictions"] = pred_vec;

    bool has_labels   = data.y.size() > 0;
    bool show_metrics = has_labels && !no_metrics;

    if (show_metrics) {
      ConfusionMatrix cm(predictions, data.y);
      result["error_rate"]       = cm.error();
      result["confusion_matrix"] = serialization::to_json(cm);
    }

    return result;
  }
}

  int run_predict(CLIOptions& params) {
    io::Output out(params.quiet);

    // Validate output path before doing work
    if (!params.output_path.empty()) {
      io::check_file_not_exists(params.output_path);
    }

    DataPacket data = [&]() {
      try {
        return io::csv::read_sorted(params.data_path);
      } catch (const ppforest2::UserError& e) {
        fmt::print(stderr, "{} {}\n", error("Error:"), e.what());
        fmt::print(stderr, "File: {}\n", params.data_path);
        std::exit(1);
      } catch (const std::exception& e) {
        fmt::print(stderr, "{} reading CSV file: {}\n", error("Error:"), e.what());
        std::exit(1);
      }
    }();

    json model_data  = io::json::read_file(params.model_path);
    auto model       = serialization::model_from_json(model_data);
    auto predictions = model->predict(data.x);

    bool has_labels   = data.y.size() > 0;
    bool show_metrics = has_labels && !params.no_metrics;

    // Terminal output: only metrics
    if (show_metrics) {
      ConfusionMatrix cm(predictions, data.y);
      out.newline();
      out.println("{}{:.2f}%", emphasis("Error rate: "), cm.error() * 100);
      out.newline();
      print_confusion_matrix(out, cm);
    }

    // Hint about --output when not used
    if (show_metrics && params.output_path.empty()) {
      out.println("{}", muted("Tip: use --output <file> to save individual predictions"));
    }

    // Save results to file if requested
    if (!params.output_path.empty()) {
      json file_result = build_predict_result(predictions, data, params.no_metrics);
      io::json::write_file(file_result, params.output_path);
      out.saved("Results", params.output_path);
    }

    return 0;
  }
}
