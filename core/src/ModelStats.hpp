#pragma once

#include <iostream>
#include <cmath>

#include <nlohmann/json.hpp>

#include "Data.hpp"

namespace pptree {
  struct ModelStats {
    stats::DataColumn<double> tr_times;
    stats::DataColumn<double> tr_error;
    stats::DataColumn<double> te_error;

    double mean_time() const {
      return tr_times.mean();
    }

    double mean_tr_error() const {
      return tr_error.mean();
    }

    double mean_te_error() const {
      return te_error.mean();
    }

    double std_time() const {
      return sd(tr_times);
    }

    double std_tr_error() const {
      return sd(tr_error);
    }

    double std_te_error() const {
      return sd(te_error);
    }

    nlohmann::json to_json() const {
      return nlohmann::json{
        { "runs",             tr_times.size() },
        { "mean_time_ms",     mean_time() },
        { "std_time_ms",      std_time() },
        { "mean_train_error", mean_tr_error() },
        { "std_train_error",  std_tr_error() },
        { "mean_test_error",  mean_te_error() },
        { "std_test_error",   std_te_error() }
      };
    }
  };

  void announce_results(const ModelStats& stats) {
    std::cout << "Evaluation results (" << stats.tr_times.size() << " runs):" << std::endl
              << "Training Time: " << stats.mean_time() << "ms ± " << stats.std_time() << "ms" << std::endl
              << "Train Error:   " << (stats.mean_tr_error() * 100) << "% ± " << (stats.std_tr_error() * 100) << "%" << std::endl
              << "Test Error:    " << (stats.mean_te_error() * 100) << "% ± " << (stats.std_te_error() * 100) << "%" << std::endl;
  }

  void announce_results_json(const ModelStats& stats) {
    std::cout << stats.to_json().dump(2) << std::endl;
  }
}
