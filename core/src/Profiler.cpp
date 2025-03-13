#include "pptree.hpp"
#include "csv.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>  // for std::iota

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
  float lambda = -1;
  int threads = -1;
  int seed = -1;
  float var_proportion = 0.5;
  float train_proportion = 0.7;
  std::string data = "";
};

CLIParams parse_args(int argc, char *argv[]) {
  CLIParams params;
  bool has_simulate = false;
  bool has_data = false;

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

    if (name == "simulate") {
      has_simulate = true;
      // Parse NxMxK format
      size_t pos1 = value.find('x');
      size_t pos2 = value.find('x', pos1 + 1);

      if (pos1 == std::string::npos || pos2 == std::string::npos) {
        std::cerr << "Invalid simulate format: " << value << "\n"
                  << "Expected format: NxMxK (rows x cols x classes)" << std::endl;
        exit(1);
      }

      params.rows = std::stoi(value.substr(0, pos1));
      params.cols = std::stoi(value.substr(pos1 + 1, pos2 - pos1 - 1));
      params.classes = std::stoi(value.substr(pos2 + 1));
    } else if (name == "data") {
      has_data = true;
      params.data = value;
    } else if (name == "trees") params.trees = std::stoi(value);
    else if (name == "lambda") {
      float lambda_val = std::stof(value);

      if (lambda_val < 0 || lambda_val > 1) {
        std::cout << "Warning: lambda clamped to [0,1] range" << std::endl;
        lambda_val = std::clamp(lambda_val, 0.0f, 1.0f);
      }

      params.lambda = lambda_val;
    } else if (name == "threads") params.threads = std::stoi(value);
    else if (name == "seed") params.seed = std::stoi(value);
    else if (name == "var-proportion") {
      float prop = std::stof(value);
      params.var_proportion = std::clamp(prop, 0.0f, 1.0f);

      if (prop != params.var_proportion) {
        std::cout << "Warning: var-proportion clamped to [0,1] range: " << params.var_proportion << std::endl;
      }
    } else if (name == "train-proportion") {
      float prop = std::stof(value);

      if (prop <= 0 || prop >= 1) {
        std::cerr << "Error: train-proportion must be in range (0,1)" << std::endl;
        exit(1);
      }

      params.train_proportion = prop;
    } else {
      std::cerr << "Unknown parameter: " << name << std::endl;
      exit(1);
    }
  }

  // Validate that exactly one of simulate or data is specified
  if (!has_simulate && !has_data) {
    std::cerr << "Error: Must specify either simulate=NxMxK or data=path" << std::endl;
    exit(1);
  }

  if (has_simulate && has_data) {
    std::cerr << "Error: Cannot specify both simulate and data parameters" << std::endl;
    exit(1);
  }

  return params;
}

SortedDataSpec<float, int> read_data(const CLIParams& params) {
  if (!params.data.empty()) {
    return parse_csv(params.data);
  } else {
    return simulate(params.rows, params.cols, params.classes);
  }
}

// Function declaration first
std::pair<SortedDataSpec<float, int>, SortedDataSpec<float, int> >
split_data(const SortedDataSpec<float, int>& data, float train_ratio, int seed);

// Function definition without default arguments
std::pair<SortedDataSpec<float, int>, SortedDataSpec<float, int> >
split_data(const SortedDataSpec<float, int>& data, float train_ratio, int seed) {
  const int n = data.x.rows();
  const int train_size = static_cast<int>(n * train_ratio);

  // Create index vector and shuffle it
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 gen(seed);
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
void evaluate_model(const Model& model, const SortedDataSpec<float, int>& test_data,
 const std::chrono::high_resolution_clock::time_point& start) {
  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "Training completed in " << duration.count() << "ms" << std::endl;
  std::cout << "Test Accuracy: " << (100 * (1 - model.error_rate(test_data))) << "%" << std::endl;
}

void print_usage(const char *program_name) {
  std::cerr << "Usage: " << program_name << " [parameter=value ...]\n"
            << "Parameters:\n"
            << "  simulate=NxMxK      - Simulate NxM data matrix with K classes\n"
            << "  trees=N             - Number of trees (0 for single tree, >0 for forest)\n"
            << "  lambda=N            - Method selection  (0=LDA, (0,1]=PDA, default: 0.5)\n"
            << "  threads=N           - Number of threads for forest training (default: number of cores)\n"
            << "  seed=N              - Random seed (default: random)\n"
            << "  var-proportion=F    - Proportion of features to use in forest (default: 0.5)\n"
            << "  train-proportion=F  - Train set proportion (default: 0.7)\n"
            << "  data=path           - CSV file to read (enables CSV mode)\n"
            << "\nExample:\n"
            << "  Forest:     " << program_name << " data=data.csv trees=500\n"
            << "  Single tree:" << program_name << " data=data.csv trees=0" << std::endl;
}

void warn_unused_params(const CLIParams& params) {
  if (params.trees == 0) {
    bool has_warnings = false;

    if (params.threads != -1) {
      std::cout << "Warning: threads parameter is ignored when training a single tree" << std::endl;
      has_warnings = true;
    }

    if (params.var_proportion != 0.5) {
      std::cout << "Warning: var-proportion parameter is ignored when training a single tree" << std::endl;
      has_warnings = true;
    }

    if (has_warnings) {
      std::cout << "Single trees always use all features for splitting" << std::endl;
    }
  }
}

void print_model_type(const CLIParams& params) {
  if (params.trees > 0) {
    std::cout << "Training random forest with " << params.trees << " trees"
              << " (using " << (params.var_proportion * 100) << "% of features per split)" << std::endl;
  } else {
    std::cout << "Training single decision tree (using all features)" << std::endl;
  }
}

void initialize_default_params(CLIParams& params) {
  if (params.lambda == -1) {
    params.lambda = 0.5;
    std::cout << "Using default lambda: " << params.lambda << std::endl;
  }

  if (params.seed == -1) {
    std::random_device rd;
    params.seed = rd();
    std::cout << "Using random seed: " << params.seed << std::endl;
  }

  if (params.threads == -1) {
        #ifdef _OPENMP
    params.threads = omp_get_max_threads();
        #else
    params.threads = 1;
        #endif
    std::cout << "Using default thread count: " << params.threads << std::endl;
  }
}

void print_parameters(const CLIParams& params) {
  std::cout << "Using " << (params.lambda == 0 ? "LDA" : "PDA")
            << " (lambda=" << params.lambda << ")" << std::endl;
}

void print_split_info(const SortedDataSpec<float, int>& train_data,
 const SortedDataSpec<float, int>&                      test_data,
 float                                                  train_proportion) {
  std::cout << "Data split into:\n"
            << "Training set: " << train_data.x.rows() << " samples ("
            << (train_proportion * 100) << "%)\n"
            << "Test set:     " << test_data.x.rows() << " samples ("
            << ((1 - train_proportion) * 100) << "%)" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  // Parse and initialize parameters
  CLIParams params = parse_args(argc, argv);
  initialize_default_params(params);
  warn_unused_params(params);
  print_model_type(params);
  print_parameters(params);

  // Load and split data
  auto full_data = read_data(params);
  auto [train_data, test_data] = split_data(full_data, params.train_proportion, params.seed);
  print_split_info(train_data, test_data, params.train_proportion);

  const auto start = std::chrono::high_resolution_clock::now();

  if (params.trees > 0) {
    // Forest: use uniform_glda with var_proportion
    const auto spec = TrainingSpec<float, int>::uniform_glda(
      std::round(train_data.x.cols() * params.var_proportion),
      params.lambda
      );
    auto forest = Forest<float, int>::train(*spec, train_data, params.trees, params.seed, params.threads);
    evaluate_model(forest, test_data, start);
  } else {
    // Single tree: use regular glda without feature subsampling
    const auto spec = TrainingSpec<float, int>::glda(params.lambda);
    auto tree = Tree<float, int>::train(*spec, train_data);
    evaluate_model(tree, test_data, start);
  }

  return 0;
}
