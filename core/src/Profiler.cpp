#include "pptree.hpp"
#include "csv.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>

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

struct CLIParams {
  int rows = 100;
  int cols = 10;
  int classes = 2;
  int trees = 100;
  int lambda = 10;
  int threads = 1;
  std::string filename = "";
};

CLIParams parse_args(int argc, char *argv[]) {
  CLIParams params;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    size_t pos = arg.find('=');

    if (pos == std::string::npos) {
      std::cerr << "Invalid argument format: " << arg << "\n"
                << "Expected format: name=value" << std::endl;
      exit(1);
    }

    std::string name = arg.substr(0, pos);
    std::string value = arg.substr(pos + 1);

    if (name == "rows") params.rows = std::stoi(value);
    else if (name == "cols") params.cols = std::stoi(value);
    else if (name == "classes") params.classes = std::stoi(value);
    else if (name == "trees") params.trees = std::stoi(value);
    else if (name == "lambda") params.lambda = std::stoi(value);
    else if (name == "threads") params.threads = std::stoi(value);
    else if (name == "filename") params.filename = value;
    else {
      std::cerr << "Unknown parameter: " << name << std::endl;
      exit(1);
    }
  }

  return params;
}

SortedDataSpec<float, int> read_data(const CLIParams& params) {
  if (!params.filename.empty()) {
    return parse_csv(params.filename);
  } else {
    return simulate(params.rows, params.cols, params.classes);
  }
}

std::pair<SortedDataSpec<float, int>, SortedDataSpec<float, int> >
split_data(const SortedDataSpec<float, int>& data, float train_ratio = 0.7) {
  const int n = data.x.rows();
  const int train_size = static_cast<int>(n * train_ratio);

  // Create index vector and shuffle it
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(indices.begin(), indices.end(), gen);

  // Create train and test matrices
  Data<float> train_x(train_size, data.x.cols());
  DataColumn<int> train_y(train_size);
  Data<float> test_x(n - train_size, data.x.cols());
  DataColumn<int> test_y(n - train_size);

  // Fill train data
  for (int i = 0; i < train_size; ++i) {
    for (int j = 0; j < data.x.cols(); ++j) {
      train_x(i, j) = data.x(indices[i], j);
    }

    train_y[i] = data.y[indices[i]];
  }

  // Fill test data
  for (int i = 0; i < n - train_size; ++i) {
    for (int j = 0; j < data.x.cols(); ++j) {
      test_x(i, j) = data.x(indices[i + train_size], j);
    }

    test_y[i] = data.y[indices[i + train_size]];
  }

  return {
    SortedDataSpec<float, int>(train_x, train_y),
    SortedDataSpec<float, int>(test_x, test_y)
  };
}

template<typename Model>
void evaluate_model(const Model&                             model,
 const SortedDataSpec<float, int>&                           test_data,
 std::chrono::time_point<std::chrono::high_resolution_clock> start_time) {
  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<float> >(end - start_time).count();

  // Calculate accuracy on test set
  int correct = 0;
  for (int i = 0; i < test_data.x.rows(); ++i) {
    // Create a single-row Data matrix for prediction
    Data<float> single_row(1, test_data.x.cols());
    for (int j = 0; j < test_data.x.cols(); ++j) {
      single_row(0, j) = test_data.x(i, j);
    }

    if (model.predict(single_row)(0) == test_data.y[i]) {
      correct++;
    }
  }

  float accuracy = static_cast<float>(correct) / test_data.x.rows();

  std::cout << "Training Time: " << elapsed_time << " seconds\n"
            << "Test Accuracy: " << (accuracy * 100) << "%" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [parameter=value ...]\n"
              << "Parameters:\n"
              << "  rows=N        - Number of samples (simulation mode only)\n"
              << "  cols=N        - Number of features (simulation mode only)\n"
              << "  classes=N     - Number of classes (simulation mode only)\n"
              << "  trees=N       - Number of trees (0 for single tree)\n"
              << "  lambda=N      - Minimum leaf size\n"
              << "  threads=N     - Number of threads\n"
              << "  filename=path - CSV file to read (enables CSV mode)\n"
              << "\nExample:\n"
              << "  Simulation: " << argv[0] << " rows=100 cols=10 classes=2 trees=500\n"
              << "  CSV file:   " << argv[0] << " filename=data.csv trees=500" << std::endl;
    return 1;
  }

  CLIParams params = parse_args(argc, argv);
  auto full_data = read_data(params);

  // Split data into train and test sets
  auto [train_data, test_data] = split_data(full_data);

  std::cout << "Data split into:\n"
            << "Training set: " << train_data.x.rows() << " samples\n"
            << "Test set:     " << test_data.x.rows() << " samples\n";

  const auto spec = TrainingSpec<float, int>::uniform_glda(
    std::round(train_data.x.cols() / 2.0),
    params.lambda
    );

  const auto start = std::chrono::high_resolution_clock::now();

  // Train on training data
  if (params.trees > 0) {
    auto forest = Forest<float, int>::train(*spec, train_data, params.trees, 0, params.threads);
    evaluate_model(forest, test_data, start);
  } else {
    auto tree = Tree<float, int>::train(*spec, train_data);
    evaluate_model(tree, test_data, start);
  }

  return 0;
}
