#include "pptree.hpp"
#include "csv.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>

#include "DataPacket.hpp"
#include "Normal.hpp"

#include "CLIOptions.hpp"
#include "IO.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace pptree;
using namespace csv;

struct SimulationParams {
  float mean            = 100.0f;
  float mean_separation = 50.0f;
  float sd              = 10.0f;
};

inline DataPacket<float, int> simulate(
  const int               n,
  const int               p,
  const int               G,
  const SimulationParams& params = SimulationParams{}) {
  Data<float> x(n, p);
  DataColumn<int> y(n);

  for (int i = 0; i < n; ++i) {
    float group_mean = params.mean + (i % G) * params.mean_separation;

    Normal norm(group_mean, params.sd);

    for (int j = 0; j < p; ++j) {
      x(i, j) = norm();
    }

    y[i] = i % G;
  }

  return DataPacket<float, int>(x, y);
}

struct Split {
  std::vector<int> tr;
  std::vector<int> te;
};

inline Split split(const DataPacket<float, int>& data, float train_ratio) {
  const int n          = data.x.rows();
  const int train_size = static_cast<int>(n * train_ratio);

  DataSpec<float, int> spec(data.x, data.y);

  std::vector<int> train_indices;
  std::vector<int> test_indices;

  train_indices.reserve(train_size);
  test_indices.reserve(n - train_size);

  LOG_INFO << "Splitting data of size " << n << " into " << train_size << " training and " << n - train_size << " test samples:";
  LOG_INFO << "Classes: " << data.classes << std::endl;
  LOG_INFO << "X: " << std::endl << data.x << std::endl;
  LOG_INFO << "Y: " << std::endl << data.y << std::endl;

  for (const auto& group : data.classes) {
    int group_start      = spec.group_start(group);
    int group_size       = spec.group_size(group);
    int group_end        = group_start + group_size - 1;
    int group_train_size = static_cast<int>(group_size * train_ratio);

    LOG_INFO << "Group " << group
             << ": start=" << group_start
             << ": end=" << group_end
             << ", size=" << group_size
             << ", train_size=" << group_train_size << std::endl;

    Uniform unif(group_start, group_end);
    std::vector<int> group_indices = unif.distinct(group_size);

    train_indices.insert(train_indices.end(), group_indices.begin(), group_indices.begin() + group_train_size);
    test_indices.insert(test_indices.end(), group_indices.begin() + group_train_size, group_indices.end());
  }

  LOG_INFO << "Final split has " << train_indices.size() << " training and " << test_indices.size() << " test samples:";

  return {
    .tr = train_indices,
    .te = test_indices
  };
}

// Display a progress bar on the console
void display_progress(int current, int total, int bar_width = 50) {
  float progress = static_cast<float>(current) / total;
  int pos        = static_cast<int>(bar_width * progress);

  std::cout << "\r" << std::string(pos, '-') << std::string(bar_width - pos, ' ')
            << " | " << current << "/" << total
            << " (" << static_cast<int>(progress * 100.0) << "%)     " << std::flush;

  if (current == total) {
    std::cout << std::endl;
  }
}

DataPacket<float, int> read_data(const CLIOptions& params) {
  if (!params.data_path.empty()) {
    try {
      const DataPacket<float, int> data = read_csv(params.data_path);

      Data<float> x     = data.x;
      DataColumn<int> y = data.y;

      if (!DataSpec<float, int>::is_contiguous(y)) {
        models::stats::sort(x, y);
      }

      return DataPacket<float, int>(x, y);
    } catch (const std::runtime_error& e) {
      std::cerr << "Error reading CSV file: " << e.what() << std::endl;
      std::cerr << "Please ensure the file exists and is properly formatted" << std::endl;
      exit(1);
    } catch (const std::exception& e) {
      std::cerr << "Unexpected error reading file: " << e.what() << std::endl;
      exit(1);
    }
  } else {
    SimulationParams simulation_params;
    simulation_params.mean            = params.sim_mean;
    simulation_params.mean_separation = params.sim_mean_separation;
    simulation_params.sd              = params.sim_sd;

    try {
      return simulate(params.rows, params.cols, params.classes, simulation_params);
    } catch (const std::exception& e) {
      std::cerr << "Error simulating data: " << e.what() << std::endl;
      exit(1);
    }
  }
}

template<typename Model>
ModelStats evaluate_model(
  const TrainingSpec<float, int>& spec,
  Data<float>&                    tr_x,
  Data<float>&                    te_x,
  DataColumn<int>&                tr_y,
  DataColumn<int>&                te_y,
  const CLIOptions&               params) {
  ModelStats stats;

  stats.tr_times = DataColumn<double>(params.n_runs);
  stats.tr_error = DataColumn<double>(params.n_runs);
  stats.te_error = DataColumn<double>(params.n_runs);

  Random::seed(params.seed);

  std::cout << "Running " << params.n_runs << " iterations:" << std::endl;

  for (int i = 0; i < params.n_runs; ++i) {
    display_progress(i + 1, params.n_runs);

    const auto start = std::chrono::high_resolution_clock::now();

    const Model model = [&]() {
        if constexpr (std::is_same_v<Model, Forest<float, int> >) {
          return Forest<float, int>::train(spec, tr_x, tr_y, params.trees, params.seed + i, params.threads);
        } else {
          return Tree<float, int>::train(spec, tr_x, tr_y);
        }
      }();

    const auto end      = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    stats.tr_times[i] = duration.count();
    stats.tr_error[i] = model.error_rate(tr_x, tr_y);
    stats.te_error[i] = model.error_rate(te_x, te_y);
  }

  std::cout << std::endl;

  return stats;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  CLIOptions params = parse_args(argc, argv);

  auto full_data = read_data(params);

  init_params(params, full_data.x.cols());

  Random::seed(params.seed);
  auto data_split = split(full_data, params.train_ratio);

  Data<float> tr_x     = full_data.x(data_split.tr, Eigen::all);
  Data<float> te_x     = full_data.x(data_split.te, Eigen::all);
  DataColumn<int> tr_y = full_data.y(data_split.tr);
  DataColumn<int> te_y = full_data.y(data_split.te);

  std::cout << std::endl;

  announce_configuration(params, tr_x, te_x);

  ModelStats stats;

  if (params.trees > 0) {
    auto spec = TrainingSpecUGLDA<float, int>(params.n_vars, params.lambda);
    stats = evaluate_model<Forest<float, int> >(spec, tr_x, te_x, tr_y, te_y, params);
  } else {
    auto spec = TrainingSpecGLDA<float, int>(params.lambda);
    stats = evaluate_model<Tree<float, int> >(spec, tr_x, te_x, tr_y, te_y, params);
  }

  announce_results(stats);
  return 0;
}
