/**
 * @file Reproducibility.test.cpp
 * @brief Tests that compare freshly-trained models against committed golden files.
 *
 * Each test loads a golden JSON file, trains the same model configuration,
 * and compares predictions, error rates, confusion matrix, and (for forests)
 * OOB error and variable importance.
 */
#include <gtest/gtest.h>

#include "utils/Types.hpp"
#include "utils/Math.hpp"
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
#include "utils/Invariant.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace pptree;
using namespace pptree::types;
using namespace pptree::stats;
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

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

namespace {
  json load_golden(const std::string& path) {
    std::ifstream in(path);
    return json::parse(in);
  }

  /**
   * @brief Resolve the path to the golden file for a given dataset and slug.
   *
   * Tries to load the platform-specific golden file first, then falls back to
   * the base golden file.
   *
   * @param dataset The name of the dataset.
   * @param slug The slug of the golden file.
   * @return The path to the golden file.
   */
  std::string resolve_golden_path(const std::string& dataset, const std::string& slug) {
    std::string platform_path = GOLDEN_DIR + "/" + dataset + "/" + slug + "." + PLATFORM + ".json";

    if (std::filesystem::exists(platform_path)) {
      return platform_path;
    }

    return GOLDEN_DIR + "/" + dataset + "/" + slug + ".json";
  }

  void compare_predictions(const json& expected_json, const ResponseVector& actual) {
    auto expected = expected_json.get<std::vector<int>>();

    ASSERT_EQ(static_cast<int>(expected.size()), actual.size()) << "Prediction count mismatch";

    for (int i = 0; i < actual.size(); ++i) {
      EXPECT_EQ(expected[i], actual(i)) << "Prediction mismatch at index " << i;
    }
  }

  void compare_confusion_matrix(const json& expected_json, const ConfusionMatrix& actual) {
    auto expected_matrix = expected_json["matrix"];
    auto expected_labels = expected_json["labels"].get<std::vector<int>>();

    ASSERT_EQ(static_cast<int>(expected_matrix.size()), actual.values.rows());

    for (int i = 0; i < actual.values.rows(); ++i) {
      for (int j = 0; j < actual.values.cols(); ++j) {
        EXPECT_EQ(expected_matrix[i][j].get<int>(), actual.values(i, j)) << "Confusion matrix mismatch at (" << i << ", " << j << ")";
      }
    }

    // Compare labels
    std::vector<int> actual_labels;
    for (const auto& [label, idx] : actual.label_index) {
      actual_labels.push_back(label);
    }

    ASSERT_EQ(expected_labels, actual_labels);
  }

  void compare_vi(const json& expected_json, const std::string& key,
    const FeatureVector& actual, float tolerance) {
    if (!expected_json.contains(key)) return;

    if (actual.size() == 0) return;

    auto expected = expected_json[key].get<std::vector<float>>();

    ASSERT_EQ(static_cast<int>(expected.size()), actual.size()) << "VI '" << key << "' size mismatch";

    for (int i = 0; i < actual.size(); ++i) {
      EXPECT_NEAR(expected[i], actual(i), tolerance) << "VI '" << key << "' mismatch at index " << i;
    }
  }
} // anonymous namespace

// ---------------------------------------------------------------------------
// Macro to reduce boilerplate for each golden test
// ---------------------------------------------------------------------------

#define GOLDEN_TREE_TEST(TestName, dataset_name, slug_name, csv_file, lambda_val, seed_val) \
        TEST(Reproducibility, TestName) {                                                   \
          auto path = resolve_golden_path(dataset_name, slug_name);                         \
          invariant(std::filesystem::exists(path), "Golden file missing: " + path);            \
                                                                                            \
          auto golden = load_golden(path);                                                  \
          auto data   = io::read_csv_sorted(DATA_DIR + "/" + csv_file);                     \
                                                                                            \
          RNG rng(seed_val);                                                                \
          Tree tree = Tree::train(TrainingSpecGLDA(lambda_val), data.x, data.y, rng);       \
                                                                                            \
          auto predictions = tree.predict(data.x);                                          \
          compare_predictions(golden["predictions"], predictions);                          \
                                                                                            \
          ConfusionMatrix cm(predictions, data.y);                                          \
          EXPECT_NEAR(cm.error(), golden["error_rate"].get<float>(), 1e-3f);                \
          compare_confusion_matrix(golden["confusion_matrix"], cm);                         \
                                                                                            \
          if (golden.contains("variable_importance")) {                                     \
            const int n_vars    = static_cast<int>(data.x.cols());                          \
            FeatureVector scale = stats::sd(data.x);                                        \
            scale = (scale.array() > Feature(0)).select(scale, Feature(1));                 \
            FeatureVector vi2 = variable_importance_projections(tree, n_vars, &scale);      \
            compare_vi(golden["variable_importance"], "projections", vi2, 1e-3f);           \
          }                                                                                 \
        }

#define GOLDEN_FOREST_TEST(TestName, dataset_name, slug_name, csv_file,                         \
          n_trees, lambda_val, n_vars_val, seed_val)                                            \
        TEST(Reproducibility, TestName) {                                                       \
          auto path = resolve_golden_path(dataset_name, slug_name);                             \
          invariant(std::filesystem::exists(path), "Golden file missing: " + path);               \
                                                                                                \
          auto golden = load_golden(path);                                                      \
          auto data   = io::read_csv_sorted(DATA_DIR + "/" + csv_file);                         \
                                                                                                \
          Forest forest = Forest::train(                                                        \
            TrainingSpecUGLDA(n_vars_val, lambda_val),                                          \
            data.x, data.y, n_trees, seed_val, 1);                                              \
                                                                                                \
          auto predictions = forest.predict(data.x);                                            \
          compare_predictions(golden["predictions"], predictions);                              \
                                                                                                \
          ConfusionMatrix cm(predictions, data.y);                                              \
          EXPECT_NEAR(cm.error(), golden["error_rate"].get<float>(), 1e-3f);                    \
          compare_confusion_matrix(golden["confusion_matrix"], cm);                             \
                                                                                                \
          if (golden.contains("oob_error")) {                                                   \
            double oob_err = forest.oob_error(data.x, data.y);                                  \
            EXPECT_NEAR(oob_err, golden["oob_error"].get<double>(), 1e-3);                      \
          }                                                                                     \
                                                                                                \
          if (golden.contains("variable_importance")) {                                         \
            const int n_v       = static_cast<int>(data.x.cols());                              \
            FeatureVector scale = stats::sd(data.x);                                            \
            scale = (scale.array() > Feature(0)).select(scale, Feature(1));                     \
                                                                                                \
            FeatureVector vi1 = variable_importance_permuted(forest, data.x, data.y, seed_val); \
            FeatureVector vi2 = variable_importance_projections(forest, n_v, &scale);           \
            FeatureVector vi3 = variable_importance_weighted_projections(                       \
              forest, data.x, data.y, &scale);                                                  \
                                                                                                \
            compare_vi(golden["variable_importance"], "projections", vi2, 1e-3f);               \
            compare_vi(golden["variable_importance"], "permuted", vi1, 1e-3f);                  \
            compare_vi(golden["variable_importance"], "weighted_projections", vi3, 1e-3f);      \
          }                                                                                     \
        }

// ---------------------------------------------------------------------------
// Tree tests
// ---------------------------------------------------------------------------

GOLDEN_TREE_TEST(IrisTreeGLDA, "iris", "tree-glda-s42", "iris.csv", 0.0f, 42)
GOLDEN_TREE_TEST(CrabTreeGLDA, "crab", "tree-glda-s42", "crab.csv", 0.0f, 42)

// ---------------------------------------------------------------------------
// Forest tests
// ---------------------------------------------------------------------------

GOLDEN_FOREST_TEST(IrisForestGLDA, "iris", "forest-glda-t5-s42", "iris.csv", 5, 0.0f, 2, 42)
GOLDEN_FOREST_TEST(IrisForestPDA, "iris", "forest-pda-l05-t5-s42", "iris.csv", 5, 0.5f, 2, 42)
GOLDEN_FOREST_TEST(CrabForestGLDA, "crab", "forest-glda-t10-s42", "crab.csv", 10, 0.0f, 3, 42)
GOLDEN_FOREST_TEST(WineForestGLDA, "wine", "forest-glda-t10-s42", "wine.csv", 10, 0.0f, 4, 42)
GOLDEN_FOREST_TEST(GlassForestGLDA, "glass", "forest-glda-t10-s42", "glass.csv", 10, 0.0f, 3, 42)
