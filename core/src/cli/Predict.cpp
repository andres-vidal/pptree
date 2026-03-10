/**
 * @file Predict.cpp
 * @brief Predict subcommand handler.
 */
#include "cli/Predict.hpp"
#include "pptree.hpp"

#include <fmt/format.h>
#include <fstream>
#include <vector>

#include "stats/DataPacket.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "io/Presentation.hpp"
#include "io/Color.hpp"
#include "io/IO.hpp"
#include "serialization/Json.hpp"

#include <nlohmann/json.hpp>

using namespace pptree;
using namespace pptree::types;
using namespace pptree::stats;
using namespace pptree::io;
using json = nlohmann::json;

namespace pptree::cli {
namespace {
  json load_model(const std::string& path) {
    std::ifstream in(path);

    if (!in.is_open()) {
      fmt::print(stderr, "Error: Could not open model file: {}\n", path);
      std::exit(1);
    }

    try {
      return json::parse(in);
    } catch (const json::parse_error& e) {
      fmt::print(stderr, "Error: Invalid JSON in model file: {}\n", e.what());
      std::exit(1);
    }
  }

  json build_predict_result(
    const ResponseVector& predictions,
    const DataPacket&     data,
    bool                  no_metrics) {
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
    // Validate output path before doing work
    if (!params.output_path.empty()) {
      check_file_not_exists(params.output_path);
    }

    DataPacket data = [&]() {
      try {
        return read_csv_sorted(params.data_path);
      } catch (const std::exception& e) {
        fmt::print(stderr, "{} reading CSV file: {}\n", error("Error:"), e.what());
        std::exit(1);
      }
    }();

    json model_data  = load_model(params.model_path);
    auto model       = serialization::model_from_json(model_data);
    auto predictions = model->predict(data.x);

    bool has_labels   = data.y.size() > 0;
    bool show_metrics = has_labels && !params.no_metrics;

    // Terminal output: only metrics
    if (!params.quiet && show_metrics) {
      ConfusionMatrix cm(predictions, data.y);
      fmt::print("\n{}{:.2f}%\n\n", emphasis("Error rate: "), cm.error() * 100);
      print_confusion_matrix(cm);
    }

    // Hint about --output when not used
    if (!params.quiet && show_metrics && params.output_path.empty()) {
      fmt::print("{}\n", muted("Tip: use --output <file> to save individual predictions"));
    }

    // Save results to file if requested
    if (!params.output_path.empty()) {
      json file_result = build_predict_result(predictions, data, params.no_metrics);
      write_json_file(file_result, params.output_path);

      if (!params.quiet) {
        fmt::print("{}{}\n", success("Results saved to "), params.output_path);
      }
    }

    return 0;
  }
}
