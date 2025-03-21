#pragma once

#include "csv.hpp"
#include <vector>
#include <random>
#include <unordered_map>
#include <iostream>
#include <numeric>
#include <cmath>

namespace pptree {
  SortedDataSpec<float, int> read_csv(const std::string& filename) {
    csv::CSVReader reader(filename);
    std::vector<std::vector<float> > featureData;
    std::vector<std::string> rawLabels;

    for (csv::CSVRow& row : reader) {
      if (row.size() < 1) {
        throw std::runtime_error("CSV row has no columns.");
      }

      std::vector<float> currentFeatures;
      for (int j = 0; j < row.size() - 1; ++j) {
        currentFeatures.push_back(row[j].get<float>());
      }

      featureData.push_back(std::move(currentFeatures));

      // Read the last column as a string.
      std::string labelStr = row[row.size() - 1].get<std::string>();
      rawLabels.push_back(labelStr);
    }

    if (featureData.empty()) {
      throw std::runtime_error("CSV file is empty.");
    }

    // Map string labels to integer codes.
    std::unordered_map<std::string, int> labelMapping;
    std::vector<int> labels;
    int labelIndex = 0;
    for (const auto &labelStr : rawLabels) {
      if (labelMapping.find(labelStr) == labelMapping.end()) {
        labelMapping[labelStr] = labelIndex++;
      }

      labels.push_back(labelMapping[labelStr]);
    }

    // Determine dimensions for feature matrix.
    const int n = featureData.size();
    const int p = featureData[0].size();

    Data<float> x(n, p);
    for (int i = 0; i < n; ++i) {
      if (featureData[i].size() != p) {
        throw std::runtime_error("Inconsistent number of feature columns in CSV file.");
      }

      for (int j = 0; j < p; ++j) {
        x(i, j) = featureData[i][j];
      }
    }

    DataColumn<int> y(n);
    for (int i = 0; i < n; ++i) {
      y[i] = labels[i];
    }

    return SortedDataSpec<float, int>(x, y);
  }

  void announce_configuration(
    const ProfilerOptions&            params,
    const SortedDataSpec<float, int>& train_data,
    const SortedDataSpec<float, int>& test_data) {
    if (params.trees > 0) {
      std::cout << "Training random forest with " << params.trees << " trees" << std::endl;
      std::cout << "Using " << params.n_vars << " variables per split (" << (params.p_vars * 100) << "% of features)" << std::endl;
    } else {
      std::cout << "Training single decision tree (using all features)" << std::endl;
    }

    std::cout << "Using " << (params.lambda == 0 ? "LDA" : "PDA") << " (lambda=" << params.lambda << ")" << std::endl;
    std::cout << "\nData split into:\n"
              << "Training set: " << train_data.x.rows() << " samples (" << (params.train_ratio * 100) << "%)\n"
              << "Test set:     " << test_data.x.rows()  << " samples (" << ((1 - params.train_ratio) * 100) << "%)\n" << std::endl;
  }

  struct ModelStats {
    std::vector<double> train_times;
    std::vector<double> train_errors;
    std::vector<double> test_errors;

    double mean_time() const {
      return std::accumulate(train_times.begin(), train_times.end(), 0.0) / train_times.size();
    }

    double std_time() const {
      if (train_times.size() <= 1) return 0.0;

      double mean = mean_time();
      double sq_sum = std::inner_product(train_times.begin(), train_times.end(), train_times.begin(), 0.0,
          std::plus<>(), [mean](double x, double y) {
            return (x - mean) * (y - mean);
          });
      return std::sqrt(sq_sum / (train_times.size() - 1));
    }

    double mean_train_error() const {
      return std::accumulate(train_errors.begin(), train_errors.end(), 0.0) / train_errors.size();
    }

    double mean_test_error() const {
      return std::accumulate(test_errors.begin(), test_errors.end(), 0.0) / test_errors.size();
    }

    double std_train_error() const {
      if (train_errors.size() <= 1) return 0.0;

      double mean = mean_train_error();
      double sq_sum = std::inner_product(train_errors.begin(), train_errors.end(), train_errors.begin(), 0.0,
          std::plus<>(), [mean](double x, double y) {
            return (x - mean) * (y - mean);
          });
      return std::sqrt(sq_sum / (train_errors.size() - 1));
    }

    double std_test_error() const {
      if (test_errors.size() <= 1) return 0.0;

      double mean = mean_test_error();
      double sq_sum = std::inner_product(test_errors.begin(), test_errors.end(), test_errors.begin(), 0.0,
          std::plus<>(), [mean](double x, double y) {
            return (x - mean) * (y - mean);
          });
      return std::sqrt(sq_sum / (test_errors.size() - 1));
    }
  };

  void announce_results(const ModelStats& stats) {
    std::cout << "Evaluation Results (" << stats.train_times.size() << " runs):" << std::endl
              << "Training Time: " << stats.mean_time() << "ms ± " << stats.std_time() << "ms" << std::endl
              << "Train Error:   " << (stats.mean_train_error() * 100) << "% ± " << (stats.std_train_error() * 100) << "%" << std::endl
              << "Test Error:    " << (stats.mean_test_error() * 100) << "% ± " << (stats.std_test_error() * 100) << "%" << std::endl;
  }
}
