#include "pptree.hpp"
#include "csv.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

#include "ProfilerOptions.hpp"
#include "IO.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace pptree;
using namespace csv;

SortedDataSpec<float, int> read_data(const ProfilerOptions& params) {
  if (!params.data_path.empty()) {
    try {
      return read_csv(params.data_path);
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
    simulation_params.mean = params.sim_mean;
    simulation_params.mean_separation = params.sim_mean_separation;
    simulation_params.sd = params.sim_sd;

    try {
      return simulate(params.rows, params.cols, params.classes, simulation_params);
    } catch (const std::exception& e) {
      std::cerr << "Error simulating data: " << e.what() << std::endl;
      exit(1);
    }
  }
}

template<typename Model>
ModelStats evaluate_model(const TrainingSpec<float, int>& spec,
  const SortedDataSpec<float, int>&                       train_data,
  const SortedDataSpec<float, int>&                       test_data,
  const ProfilerOptions&                                  params) {
  ModelStats stats;
  stats.train_times.reserve(params.n_runs);
  stats.train_errors.reserve(params.n_runs);
  stats.test_errors.reserve(params.n_runs);

  Random::seed(params.seed);

  for (int i = 0; i < params.n_runs; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();


    const Model model = [&]() {
      if constexpr (std::is_same_v<Model, Forest<float, int> >) {
        return Forest<float, int>::train(spec, train_data, params.trees, params.seed + i, params.threads);
      } else {
        return Tree<float, int>::train(spec, train_data);
      }
    }();

    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    stats.train_times.push_back(duration.count());
    stats.train_errors.push_back(model.error_rate(train_data));
    stats.test_errors.push_back(model.error_rate(test_data));
  }

  return stats;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  ProfilerOptions params = parse_args(argc, argv);

  auto full_data = read_data(params);

  init_params(params, full_data.x.cols());

  Random::seed(params.seed);
  auto [train_data, test_data] = split(full_data, params.train_ratio);

  std::cout << std::endl;

  announce_configuration(params, train_data, test_data);

  ModelStats stats;

  if (params.trees > 0) {
    auto spec = TrainingSpec<float, int>::uniform_glda(params.n_vars, params.lambda);
    stats = evaluate_model<Forest<float, int> >(*spec, train_data, test_data, params);
  } else {
    auto spec = TrainingSpec<float, int>::glda(params.lambda);
    stats = evaluate_model<Tree<float, int> >(*spec, train_data, test_data, params);
  }

  announce_results(stats);
  return 0;
}
