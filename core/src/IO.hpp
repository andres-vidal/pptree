#pragma once

#include "csv.hpp"
#include <vector>
#include <random>
#include <unordered_map>
#include <iostream>
#include <numeric>
#include <cmath>

#include "Data.hpp"
namespace pptree {
  DataPacket<float, int> read_csv(const std::string& filename) {
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

    return DataPacket<float, int>(x, y);
  }

  void announce_configuration(
    const CLIOptions&    params,
    const Data<float  >& tr_x,
    const Data<float  >& te_x) {
    if (params.trees > 0) {
      std::cout << "Training random forest with " << params.trees << " trees" << std::endl;
      std::cout << "Using " << params.n_vars << " variables per split (" << (params.p_vars * 100) << "% of features)" << std::endl;
      std::cout << "Using " << params.threads << " threads" << std::endl;
      std::cout << "Using " << params.seed << " seed" << std::endl;
    } else {
      std::cout << "Training single decision tree (using all features)" << std::endl;
    }

    std::cout << "Using " << (params.lambda == 0 ? "LDA" : "PDA") << " (lambda=" << params.lambda << ")" << std::endl << std::endl;
    std::cout << "Data split into:" << std::endl
              << "Training set: " << tr_x.rows() << " samples (" << (params.train_ratio * 100) << "%)" << std::endl
              << "Test set:     " << te_x.rows()  << " samples (" << ((1 - params.train_ratio) * 100) << "%)" << std::endl << std::endl;
  }

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
  };

  void announce_results(const ModelStats& stats) {
    std::cout << "Evaluation results (" << stats.tr_times.size() << " runs):" << std::endl
              << "Training Time: " << stats.mean_time() << "ms ± " << stats.std_time() << "ms" << std::endl
              << "Train Error:   " << (stats.mean_tr_error() * 100) << "% ± " << (stats.std_tr_error() * 100) << "%" << std::endl
              << "Test Error:    " << (stats.mean_te_error() * 100) << "% ± " << (stats.std_te_error() * 100) << "%" << std::endl;
  }
}
