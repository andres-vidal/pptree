/**
 * @file GoldenGen.cpp
 * @brief Generates golden reference files for reproducibility testing.
 *
 * For each configuration (dataset × model params), trains a model with a
 * fixed seed, computes predictions, metrics, and variable importance, then
 * writes the results as a JSON golden file.
 */
#include "pptree.hpp"

#include "utils/Types.hpp"
#include "stats/DataPacket.hpp"
#include "stats/Stats.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "models/Tree.hpp"
#include "models/Forest.hpp"
#include "models/TrainingSpecGLDA.hpp"
#include "models/TrainingSpecUGLDA.hpp"
#include "models/VariableImportance.hpp"
#include "serialization/Json.hpp"
#include "io/IO.hpp"
#include "io/Presentation.hpp"

#include <nlohmann/json.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

using namespace pptree;
using namespace pptree::types;
using namespace pptree::stats;
using namespace pptree::io;
using namespace pptree::serialization;
using json = nlohmann::json;

#ifndef PPTREE_DATA_DIR
#error "PPTREE_DATA_DIR must be defined"
#endif

#ifndef PPTREE_GOLDEN_DIR
#error "PPTREE_GOLDEN_DIR must be defined"
#endif

#ifndef PPTREE_PLATFORM
#error "PPTREE_PLATFORM must be defined"
#endif

static const std::string DATA_DIR   = PPTREE_DATA_DIR;
static const std::string GOLDEN_DIR = PPTREE_GOLDEN_DIR;
static const std::string PLATFORM   = PPTREE_PLATFORM;

struct GoldenConfig {
  std::string csv_path;
  int trees;                // 0 = single tree
  float lambda;
  int n_vars;               // ignored for single tree
  int seed;

  std::string dataset() const {
    return std::filesystem::path(csv_path).stem().string();
  }

  std::string slug() const {
    std::string s;

    if (trees > 0) {
      s = (lambda > 0) ? "forest-pda" : "forest-glda";

      if (lambda > 0) {
        std::string l = fmt::format("{:g}", lambda);
        l.erase(std::remove(l.begin(), l.end(), '.'), l.end());
        s += fmt::format("-l{}", l);
      }

      s += fmt::format("-t{}", trees);
    } else {
      s = "tree-glda";
    }

    s += fmt::format("-s{}", seed);
    return s;
  }
};

/**
 * @brief Generate a golden file for a given configuration.
 *
 * Trains a model with the given configuration, computes predictions, metrics,
 * and variable importance, then writes the results as a JSON golden file .
 *
 * @param config The configuration to generate a golden file for.
 */
static void generate_golden(const GoldenConfig& config) {
  std::string dir  = GOLDEN_DIR + "/" + config.dataset();
  std::string path = dir + "/" + config.slug() + ".json";

  std::filesystem::create_directories(dir);

  DataPacket data = read_csv_sorted(config.csv_path);

  const int n_vars = static_cast<int>(data.x.cols());

  json result;

  // Meta
  json meta;
  meta["platform"] = PLATFORM;
  meta["seed"]     = config.seed;
  meta["dataset"]  = config.dataset();
  meta["trees"]    = config.trees;
  meta["lambda"]   = config.lambda;

  if (config.trees > 0) {
    meta["n_vars"]        = config.n_vars;
    meta["training_spec"] = "uglda";
  } else {
    meta["training_spec"] = "glda";
  }

  result["meta"] = meta;

  // Train and serialize
  if (config.trees > 0) {
    Forest forest = Forest::train(
      TrainingSpecUGLDA(config.n_vars, config.lambda),
      data.x,
      data.y,
      config.trees,
      config.seed,
      1);

    result["model"] = to_json(forest);

    ResponseVector predictions = forest.predict(data.x);
    std::vector<int> pred_vec(predictions.data(), predictions.data() + predictions.size());
    result["predictions"] = pred_vec;

    ConfusionMatrix cm(predictions, data.y);
    result["error_rate"]       = cm.error();
    result["confusion_matrix"] = to_json(cm);

    double oob_err = forest.oob_error(data.x, data.y);
    result["oob_error"] = oob_err;

    // Variable importance
    FeatureVector scale = stats::sd(data.x);
    scale = (scale.array() > Feature(0)).select(scale, Feature(1));

    FeatureVector vi1 = variable_importance_permuted(forest, data.x, data.y, config.seed);
    FeatureVector vi2 = variable_importance_projections(forest, n_vars, &scale);
    FeatureVector vi3 = variable_importance_weighted_projections(forest, data.x, data.y, &scale);

    result["variable_importance"] = cli::vi_to_json(vi1, vi2, vi3, scale);
  } else {
    // Single tree
    RNG rng(config.seed);
    Tree tree = Tree::train(TrainingSpecGLDA(config.lambda), data.x, data.y, rng);

    result["model"] = to_json(tree);

    ResponseVector predictions = tree.predict(data.x);
    std::vector<int> pred_vec(predictions.data(), predictions.data() + predictions.size());
    result["predictions"] = pred_vec;

    ConfusionMatrix cm(predictions, data.y);
    result["error_rate"]       = cm.error();
    result["confusion_matrix"] = to_json(cm);

    // VI2 only for single tree
    FeatureVector scale = stats::sd(data.x);
    scale = (scale.array() > Feature(0)).select(scale, Feature(1));

    FeatureVector vi2 = variable_importance_projections(tree, n_vars, &scale);
    FeatureVector empty;

    result["variable_importance"] = cli::vi_to_json(empty, vi2, empty, scale);
  }

  write_json_file(result, path);
  fmt::print("  {} -> {}\n", config.slug(), path);
}

int main(int argc, char *argv[]) {
  std::string output_dir = (argc > 1) ? argv[1] : GOLDEN_DIR;

  fmt::print("Generating golden files in: {}\n", output_dir);
  fmt::print("Platform: {}\n\n", PLATFORM);

  std::vector<GoldenConfig> configs = {
    { DATA_DIR + "/iris.csv",  0,  0.0f, 0, 42 },
    { DATA_DIR + "/iris.csv",  5,  0.0f, 2, 42 },
    { DATA_DIR + "/iris.csv",  5,  0.5f, 2, 42 },
    { DATA_DIR + "/crab.csv",  0,  0.0f, 0, 42 },
    { DATA_DIR + "/crab.csv",  10, 0.0f, 3, 42 },
    { DATA_DIR + "/wine.csv",  10, 0.0f, 4, 42 },
    { DATA_DIR + "/glass.csv", 10, 0.0f, 3, 42 },
  };

  for (const auto& config : configs) {
    generate_golden(config);
  }

  fmt::print("\n{}\n", success("Done. " + emphasis(std::to_string(configs.size())) + " golden files generated."));
  return 0;
}
