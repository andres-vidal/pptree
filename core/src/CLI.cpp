#include "pptree.hpp"
#include "csv.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>

#include <nlohmann/json.hpp>

#include "DataPacket.hpp"
#include "Normal.hpp"

#include "CLIOptions.hpp"
#include "IO.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace pptree;
using namespace csv;
using json = nlohmann::json;

struct SimulationParams {
  float mean            = 100.0f;
  float mean_separation = 50.0f;
  float sd              = 10.0f;
};

inline DataPacket<float, int> simulate(
  const int               n,
  const int               p,
  const int               G,
  stats::RNG&             rng,
  const SimulationParams& params = SimulationParams{}) {
  Data<float> x(n, p);
  DataColumn<int> y(n);

  for (int i = 0; i < n; ++i) {
    float group_mean = params.mean + (i % G) * params.mean_separation;

    Normal norm(group_mean, params.sd);

    for (int j = 0; j < p; ++j) {
      x(i, j) = norm(rng);
    }

    y[i] = i % G;
  }

  models::stats::sort(x, y);

  return DataPacket<float, int>(x, y);
}

struct Split {
  std::vector<int> tr;
  std::vector<int> te;
};

inline Split split(const DataPacket<float, int>& data, float train_ratio, stats::RNG& rng) {
  const int n          = data.x.rows();
  const int train_size = static_cast<int>(n * train_ratio);

  GroupSpec<int> spec(data.y);

  std::vector<int> train_indices;
  std::vector<int> test_indices;

  train_indices.reserve(train_size);
  test_indices.reserve(n - train_size);

  for (const auto& group : data.classes) {
    int group_start      = spec.group_start(group);
    int group_size       = spec.group_size(group);
    int group_end        = group_start + group_size - 1;
    int group_train_size = static_cast<int>(group_size * train_ratio);

    Uniform unif(group_start, group_end);
    std::vector<int> group_indices = unif.distinct(group_size, rng);

    train_indices.insert(train_indices.end(), group_indices.begin(), group_indices.begin() + group_train_size);
    test_indices.insert(test_indices.end(), group_indices.begin() + group_train_size, group_indices.end());
  }

  return {
    .tr = train_indices,
    .te = test_indices
  };
}

void display_progress(int current, int total, bool quiet, int bar_width = 50) {
  if (quiet) return;

  float progress = static_cast<float>(current) / total;
  int pos        = static_cast<int>(bar_width * progress);

  std::cout << "\r" << std::string(pos, '-') << std::string(bar_width - pos, ' ')
            << " | " << current << "/" << total
            << " (" << static_cast<int>(progress * 100.0) << "%)     " << std::flush;

  if (current == total) {
    std::cout << std::endl;
  }
}

DataPacket<float, int> read_data(const CLIOptions& params, stats::RNG& rng) {
  if (!params.data_path.empty()) {
    try {
      const DataPacket<float, int> data = read_csv(params.data_path);

      Data<float> x     = data.x;
      DataColumn<int> y = data.y;

      if (!GroupSpec<int>::is_contiguous(y)) {
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
      return simulate(params.rows, params.cols, params.classes, rng, simulation_params);
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
  const CLIOptions&               params,
  stats::RNG&                     rng) {
  ModelStats stats;

  stats.tr_times = DataColumn<double>(params.n_runs);
  stats.tr_error = DataColumn<double>(params.n_runs);
  stats.te_error = DataColumn<double>(params.n_runs);

  if (!params.quiet) {
    std::cout << "Running " << params.n_runs << " iterations:" << std::endl;
  }

  for (int i = 0; i < params.n_runs; ++i) {
    display_progress(i + 1, params.n_runs, params.quiet);

    const auto start = std::chrono::high_resolution_clock::now();

    const Model model = [&]() {
        if constexpr (std::is_same_v<Model, Forest<float, int> >) {
          return Forest<float, int>::train(spec, tr_x, tr_y, params.trees, params.seed + i, params.threads);
        } else {
          return Tree<float, int>::train(spec, tr_x, tr_y, rng);
        }
      }();

    const auto end      = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    stats.tr_times[i] = duration.count();
    stats.tr_error[i] = stats::error_rate(model.predict(tr_x), tr_y);
    stats.te_error[i] = stats::error_rate(model.predict(te_x), te_y);
  }

  if (!params.quiet) {
    std::cout << std::endl;
  }

  return stats;
}

template<typename Model>
Model train_model(
  const TrainingSpec<float, int>& spec,
  Data<float>&                    x,
  DataColumn<int>&                y,
  const CLIOptions&               params,
  stats::RNG&                     rng) {
  if (!params.quiet) {
    std::cout << "Training model..." << std::endl;
  }

  if constexpr (std::is_same_v<Model, Forest<float, int> >) {
    return Forest<float, int>::train(spec, x, y, params.trees, params.seed, params.threads);
  } else {
    return Tree<float, int>::train(spec, x, y, rng);
  }
}

void save_model(
  const json&        model_json,
  const std::string& model_type,
  const CLIOptions&  params,
  int                n_features) {
  json output;
  output["model_type"] = model_type;
  output["trees"]      = params.trees;
  output["lambda"]     = params.lambda;
  output["n_vars"]     = params.n_vars;
  output["n_features"] = n_features;
  output["model"]      = model_json;

  std::ofstream out(params.save_path);

  if (!out.is_open()) {
    std::cerr << "Error: Could not open file for writing: " << params.save_path << std::endl;
    std::exit(1);
  }

  out << output.dump(2);
  out.close();

  if (!params.quiet) {
    std::cout << "Model saved to " << params.save_path << std::endl;
  }
}

json load_model(const std::string& path) {
  std::ifstream in(path);

  if (!in.is_open()) {
    std::cerr << "Error: Could not open model file: " << path << std::endl;
    std::exit(1);
  }

  try {
    return json::parse(in);
  } catch (const json::parse_error& e) {
    std::cerr << "Error: Invalid JSON in model file: " << e.what() << std::endl;
    std::exit(1);
  }
}

int main(int argc, char *argv[]) {
  CLIOptions params = parse_args(argc, argv);

  #ifdef _OPENMP
  omp_set_num_threads(params.threads);
  #endif

  switch (params.subcommand) {
      case Subcommand::train: {
        stats::RNG rng(params.seed);
        auto data = read_data(params, rng);

        init_params(params, data.x.cols());

        Data<float> x     = data.x;
        DataColumn<int> y = data.y;

        if (params.trees > 0) {
          auto spec  = TrainingSpecUGLDA<float, int>(params.n_vars, params.lambda);
          auto model = train_model<Forest<float, int> >(spec, x, y, params, rng);

          if (!params.save_path.empty()) {
            save_model(model.to_json(), "forest", params, data.x.cols());
          }

          if (params.output_format == OutputFormat::json) {
            json result;
            result["model_type"] = "forest";
            result["trees"]      = params.trees;
            result["saved"]      = !params.save_path.empty();

            if (!params.save_path.empty()) {
              result["save_path"] = params.save_path;
            }

            std::cout << result.dump(2) << std::endl;
          }
        } else {
          auto spec  = TrainingSpecGLDA<float, int>(params.lambda);
          auto model = train_model<Tree<float, int> >(spec, x, y, params, rng);

          if (!params.save_path.empty()) {
            save_model(model.to_json(), "tree", params, data.x.cols());
          }

          if (params.output_format == OutputFormat::json) {
            json result;
            result["model_type"] = "tree";
            result["saved"]      = !params.save_path.empty();

            if (!params.save_path.empty()) {
              result["save_path"] = params.save_path;
            }

            std::cout << result.dump(2) << std::endl;
          }
        }

        break;
      }

      case Subcommand::predict: {
        json model_data = load_model(params.model_path);

        DataPacket<float, int> data = [&]() {
            try {
              const DataPacket<float, int> csv_data = read_csv(params.data_path);

              Data<float> x     = csv_data.x;
              DataColumn<int> y = csv_data.y;

              if (!GroupSpec<int>::is_contiguous(y)) {
                models::stats::sort(x, y);
              }

              return DataPacket<float, int>(x, y);
            } catch (const std::exception& e) {
              std::cerr << "Error reading CSV file: " << e.what() << std::endl;
              std::exit(1);
            }
          }();

        std::string model_type = model_data.value("model_type", "tree");
        json model_json        = model_data.contains("model") ? model_data["model"] : model_data;

        DataColumn<int> predictions;

        if (model_type == "forest") {
          auto model = Forest<float, int>::from_json(model_json);
          predictions = model.predict(data.x);
        } else {
          auto model = Tree<float, int>::from_json(model_json);
          predictions = model.predict(data.x);
        }

        if (params.output_format == OutputFormat::json) {
          json result;
          std::vector<int> pred_vec(predictions.data(), predictions.data() + predictions.size());
          result["predictions"] = pred_vec;

          if (data.y.size() > 0) {
            result["error_rate"] = stats::error_rate(predictions, data.y);
          }

          std::cout << result.dump(2) << std::endl;
        } else {
          for (int i = 0; i < predictions.size(); ++i) {
            std::cout << predictions[i] << std::endl;
          }

          if (data.y.size() > 0) {
            double error = stats::error_rate(predictions, data.y);
            std::cout << std::endl << "Error rate: " << (error * 100) << "%" << std::endl;
          }
        }

        break;
      }

      case Subcommand::evaluate: {
        stats::RNG rng(params.seed);
        auto full_data = read_data(params, rng);

        init_params(params, full_data.x.cols());

        auto data_split = split(full_data, params.train_ratio, rng);

        Data<float> tr_x     = full_data.x(data_split.tr, Eigen::all);
        Data<float> te_x     = full_data.x(data_split.te, Eigen::all);
        DataColumn<int> tr_y = full_data.y(data_split.tr);
        DataColumn<int> te_y = full_data.y(data_split.te);

        if (!params.quiet) {
          std::cout << std::endl;
        }

        announce_configuration(params, tr_x, te_x);

        ModelStats model_stats;

        if (params.trees > 0) {
          auto spec = TrainingSpecUGLDA<float, int>(params.n_vars, params.lambda);
          model_stats = evaluate_model<Forest<float, int> >(spec, tr_x, te_x, tr_y, te_y, params, rng);
        } else {
          auto spec = TrainingSpecGLDA<float, int>(params.lambda);
          model_stats = evaluate_model<Tree<float, int> >(spec, tr_x, te_x, tr_y, te_y, params, rng);
        }

        if (params.output_format == OutputFormat::json) {
          announce_results_json(model_stats);
        } else {
          announce_results(model_stats);
        }

        break;
      }

      default:
        std::cerr << "Error: No subcommand specified" << std::endl;
        return 1;
  }

  return 0;
}
