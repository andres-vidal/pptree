/**
 * @file Evaluate.cpp
 * @brief Multi-iteration model evaluation with convergence and the
 *        evaluate subcommand handler.
 */
#include "cli/Evaluate.hpp"
#include "cli/Train.hpp"
#include "pptree.hpp"

#include <fmt/format.h>
#include <cmath>
#include <vector>
#include <filesystem>
#include <Eigen/Dense>

#include "stats/DataPacket.hpp"
#include "stats/Simulation.hpp"
#include "io/Presentation.hpp"
#include "io/Color.hpp"
#include "io/IO.hpp"
#include "io/Output.hpp"
#include "io/Timing.hpp"
#include "serialization/Json.hpp"

#include <nlohmann/json.hpp>

using namespace pptree;
using namespace pptree::types;
using namespace pptree::stats;
using namespace pptree::io;
using json = nlohmann::json;

namespace pptree::cli {
namespace {
  /** @brief Result of an evaluate run containing aggregated statistics. */
  struct EvaluateResult {
    ModelStats stats;
  };

  void display_progress(int current, int total, bool quiet, int bar_width = 50) {
    if (quiet) return;

    float progress = static_cast<float>(current) / total;
    int pos        = static_cast<int>(bar_width * progress);

    std::string bar_template = emphasis("\r  {} |");
    std::string bar          = std::string(pos, '-') + std::string(bar_width - pos, ' ');

    if (current == total) {
      bar_template = success(bar_template);
    } else {
      bar_template = info(bar_template);
    }

    fmt::print(bar_template + " {}/{} ({}%)     ", bar, current, total, static_cast<int>(progress * 100.0));
    std::fflush(stdout);

    if (current == total) {
      fmt::print("\n");
    }
  }

  bool check_convergence(
    const std::vector<float>& times,
    int min_iters,
    float cv_threshold) {
    int n = static_cast<int>(times.size());

    if (n < min_iters) return false;

    double sum  = 0;
    double sum2 = 0;

    for (auto t : times) {
      sum  += t;
      sum2 += t * t;
    }

    double mean = sum / n;

    if (mean <= 0) return true;

    double variance = (sum2 / n) - (mean * mean);

    if (variance < 0) variance = 0;

    double cv = std::sqrt(variance) / mean;

    return cv < cv_threshold;
  }

  EvaluateResult evaluate_model(
    FeatureMatrix &      tr_x,
    FeatureMatrix &      te_x,
    ResponseVector &     tr_y,
    ResponseVector &     te_y,
    const CLIOptions&   params,
    pptree::stats::RNG& rng) {
    Output out(params.quiet);

    // Run warmup iterations (discarded)
    if (params.warmup > 0) {
      out.print("  Running {} warmup iterations:\n", emphasis(std::to_string(params.warmup)));
    }

    for (int i = 0; i < params.warmup; ++i) {
      display_progress(i, params.warmup, params.quiet);
      train_model(tr_x, tr_y, params, rng);
      display_progress(i + 1, params.warmup, params.quiet);
    }

    if (params.warmup > 0) {
      out.print("\n");
    }

    // Determine iteration mode
    int max_iters = params.converge ? params.max_iterations : params.iterations;

    if (params.converge) {
      out.print("  Running up to {} iterations (converge at CV < {:.0f}%):\n",
      emphasis(std::to_string(max_iters)), params.cv_threshold * 100);
    } else {
      out.print("  Running {} iterations:\n", emphasis(std::to_string(max_iters)));
    }

    // Measured iterations
    std::vector<float> times;
    std::vector<float> tr_errors;
    std::vector<float> te_errors;

    int stable_count   = 0;
    int iterations_run = 0;

    for (int i = 0; i < max_iters; ++i) {
      display_progress(i, max_iters, params.quiet);

      const auto train_result = train_model(tr_x, tr_y, params, rng);

      times.push_back(static_cast<float>(train_result.duration));
      tr_errors.push_back(pptree::stats::error_rate(train_result.model->predict(tr_x), tr_y));
      te_errors.push_back(pptree::stats::error_rate(train_result.model->predict(te_x), te_y));

      iterations_run = i + 1;
      display_progress(iterations_run, max_iters, params.quiet);

      // Check convergence
      if (params.converge &&
          check_convergence(times, params.min_iterations, params.cv_threshold) &&
          check_convergence(tr_errors, params.min_iterations, params.cv_threshold) &&
          check_convergence(te_errors, params.min_iterations, params.cv_threshold)) {
        stable_count++;

        if (stable_count >= params.stable_window) {
          display_progress(iterations_run, iterations_run, params.quiet);
          out.print("\n");
          out.print("  Converged after {} iterations (CV < {:.0f}%)\n", iterations_run, params.cv_threshold * 100);

          break;
        }
      } else if (params.converge) {
        stable_count = 0;
      }
    }

    if (params.converge && stable_count < params.stable_window) {
      out.print("\n");
      out.print("{} Did not converge after {} iterations\n", warning("Warning:"), iterations_run);
    } else {
      out.print("\n");
    }

    // Build ModelStats from collected vectors
    ModelStats model_stats;
    model_stats.tr_times = Eigen::Map<const Vector<float>>(times.data(), times.size());
    model_stats.tr_error = Eigen::Map<const Vector<float>>(tr_errors.data(), tr_errors.size());
    model_stats.te_error = Eigen::Map<const Vector<float>>(te_errors.data(), te_errors.size());

    return { model_stats };
  }

  json build_export_config(const CLIOptions& params) {
    json config;

    config["trees"]   = params.trees;
    config["lambda"]  = params.lambda;
    config["seed"]    = params.seed;
    config["threads"] = params.threads;

    if (params.trees > 0 && params.n_vars > 0) {
      config["vars"] = params.n_vars;
    }

    config["train-ratio"] = params.train_ratio;

    if (params.converge) {
      config["max-iterations"] = params.max_iterations;
      config["cv-threshold"]   = params.cv_threshold;
    } else {
      config["iterations"] = params.iterations;
    }

    config["data"] = "data.csv";

    return config;
  }

  void export_experiment(
    const CLIOptions&     params,
    const DataPacket&     full_data,
    const EvaluateResult& eval_result) {
    std::string dir = params.export_path;

    std::filesystem::create_directories(dir);

    json config = build_export_config(params);
    write_json_file(config, dir + "/config.json");

    write_csv(full_data, dir + "/data.csv");

    write_json_file(eval_result.stats.to_json(), dir + "/results.json");

    Output out(params.quiet);
    out.print("{}{}/\n", success("Experiment exported to "), dir);
  }
}

  int run_evaluate(CLIOptions& params) {
    // Validate output paths before doing work
    if (!params.output_path.empty()) {
      check_file_not_exists(params.output_path);
    }

    if (!params.export_path.empty()) {
      check_dir_not_exists(params.export_path);
    }

    pptree::stats::RNG rng(params.seed);
    auto full_data = read_data(params, rng);

    init_params(params, full_data.x.cols());

    auto data_split = split(full_data, params.train_ratio, rng);

    FeatureMatrix tr_x  = full_data.x(data_split.tr, Eigen::all);
    FeatureMatrix te_x  = full_data.x(data_split.te, Eigen::all);
    ResponseVector tr_y = full_data.y(data_split.tr);
    ResponseVector te_y = full_data.y(data_split.te);

    Output out(params.quiet);

    print_configuration(params, tr_x.rows(), te_x.rows());

    auto eval_result = evaluate_model(tr_x, te_x, tr_y, te_y, params, rng);

    // Measure peak RSS after evaluation
    eval_result.stats.peak_rss_bytes = get_peak_rss_bytes();

    if (!out.quiet) {
      print_results(eval_result.stats);
    }

    // Save results to file if requested
    if (!params.output_path.empty()) {
      write_json_file(eval_result.stats.to_json(), params.output_path);
      out.saved("Results", params.output_path);
    }

    // Export experiment bundle if requested
    if (!params.export_path.empty()) {
      export_experiment(params, full_data, eval_result);
    }

    return 0;
  }
}
