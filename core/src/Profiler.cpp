#include "pptree.hpp"
#include "csv.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace pptree;
using namespace csv;

SortedDataSpec<float, int> simulate(
  const int n,
  const int p,
  const int G) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> norm(100, 10);

  Data<float> x(n, p);

  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.cols(); ++j) {
      x(i, j) = norm(gen);
    }
  }

  DataColumn<int> y(n);

  for (int i = 0; i < y.size(); ++i) {
    y[i] = i % G;
  }

  return SortedDataSpec<float, int>(x, y);
}

SortedDataSpec<float, int> parse_csv(const std::string& filename) {
  csv::CSVReader reader(filename);
  std::vector<std::vector<float> > featureData;
  std::vector<std::string> rawLabels;

  // Process each row:
  // - All columns except the last are features.
  // - The last column is read as a string label.
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

SortedDataSpec<float, int> read_data(int argc, char *argv[]) {
  if (argc != 7 && argc != 5) {
    std::cerr << "Usage:\n"
              << "  Simulation mode: " << argv[0] << " n p G B l C\n"
              << "  CSV mode: " << argv[0] << " B l C F" << std::endl;

    exit(1);
  }

  if (argc == 7) {
    const int n = std::stoi(argv[1]);
    const int p = std::stoi(argv[2]);
    const int G = std::stoi(argv[3]);

    return simulate(n, p, G);
  } else {
    std::string F = argv[4];

    return parse_csv(F);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 7 && argc != 5) {
    std::cerr << "Usage:\n"
              << "  Simulation mode: " << argv[0] << " n p G B l C\n"
              << "  CSV mode: " << argv[0] << " B l C F" << std::endl;
    return 1;
  }

  const int B = std::stoi(argv[1]);
  const int l = std::stoi(argv[2]);
  const int C = std::stoi(argv[3]);


  SortedDataSpec<float, int> data = read_data(argc, argv);

  const auto spec = TrainingSpec<float, int>::uniform_glda(std::round(data.x.cols() / 2.0), l);

  const auto start = std::chrono::high_resolution_clock::now();

  if (B > 0) {
    Forest<float, int>::train(*spec, data, B, 0, C);
  } else {
    Tree<float, int>::train(*spec, data);
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<float> >(end - start).count();

  std::cout << "Elapsed Time: " << elapsed_time << " seconds." << std::endl;

  return 0;
}
