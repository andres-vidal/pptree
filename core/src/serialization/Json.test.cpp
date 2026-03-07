#include <gtest/gtest.h>

#include "pptree.hpp"
#include "serialization/Json.hpp"
#include "stats/Simulation.hpp"
#include "utils/Macros.hpp"

#include <fstream>

using namespace pptree;
using namespace pptree::types;
using namespace pptree::stats;
using namespace pptree::serialization;

using json = nlohmann::json;

TEST(JsonRoundTrip, Tree) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 100.0f;
  params.sd = 5.0f;
  auto data = simulate(90, 4, 3, rng, params);

  RNG train_rng(42);
  auto tree = Tree::train(TrainingSpecGLDA(0.0f), data.x, data.y, train_rng);

  json j = to_json(tree);
  Tree restored = tree_from_json(j);

  auto predictions_original = tree.predict(data.x);
  auto predictions_restored = restored.predict(data.x);

  ASSERT_EQ(predictions_original, predictions_restored);
}

TEST(JsonRoundTrip, Forest) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 100.0f;
  params.sd = 5.0f;
  auto data = simulate(90, 4, 3, rng, params);

  auto forest = Forest::train(
    TrainingSpecUGLDA(2, 0.0f),
    data.x, data.y, 5, 42, 1);

  json j = to_json(forest);
  Forest restored = forest_from_json(j);

  auto predictions_original = forest.predict(data.x);
  auto predictions_restored = restored.predict(data.x);

  ASSERT_EQ(predictions_original, predictions_restored);
}

TEST(JsonRoundTrip, ModelDispatchTree) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 100.0f;
  params.sd = 5.0f;
  auto data = simulate(90, 4, 3, rng, params);

  RNG train_rng(42);
  auto tree = Tree::train(TrainingSpecGLDA(0.0f), data.x, data.y, train_rng);

  json j = to_json(static_cast<const Model&>(tree));
  ASSERT_EQ(j["model_type"], "tree");

  auto restored = model_from_json(j);
  auto predictions_original = tree.predict(data.x);
  auto predictions_restored = restored->predict(data.x);

  ASSERT_EQ(predictions_original, predictions_restored);
}

TEST(JsonRoundTrip, ModelDispatchForest) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 100.0f;
  params.sd = 5.0f;
  auto data = simulate(90, 4, 3, rng, params);

  auto forest = Forest::train(
    TrainingSpecUGLDA(2, 0.0f),
    data.x, data.y, 5, 42, 1);

  json j = to_json(static_cast<const Model&>(forest));
  ASSERT_EQ(j["model_type"], "forest");

  auto restored = model_from_json(j);
  auto predictions_original = forest.predict(data.x);
  auto predictions_restored = restored->predict(data.x);

  ASSERT_EQ(predictions_original, predictions_restored);
}

TEST(JsonRoundTrip, ConfusionMatrix) {
  ResponseVector predictions(6);
  predictions << 0, 0, 1, 1, 2, 2;

  ResponseVector actual(6);
  actual << 0, 1, 1, 1, 2, 0;

  ConfusionMatrix cm(predictions, actual);
  json j = to_json(cm);

  ASSERT_TRUE(j.contains("matrix"));
  ASSERT_TRUE(j.contains("labels"));
  ASSERT_TRUE(j.contains("class_errors"));

  auto matrix = j["matrix"].get<std::vector<std::vector<int>>>();
  ASSERT_EQ(matrix.size(), 3);
  ASSERT_EQ(matrix[0].size(), 3);
}

TEST(JsonRoundTrip, TreePreservesStructure) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 200.0f;
  params.sd = 1.0f;
  auto data = simulate(60, 2, 2, rng, params);

  RNG train_rng(42);
  auto tree = Tree::train(TrainingSpecGLDA(0.0f), data.x, data.y, train_rng);

  json j1 = to_json(tree);
  Tree restored = tree_from_json(j1);
  json j2 = to_json(restored);

  ASSERT_EQ(j1, j2) << "JSON should be identical after round-trip";
}

TEST(JsonRoundTrip, FromModelJsonFile) {
  std::string model_path = std::string(PPTREE_DATA_DIR) + "/../model.json";
  std::ifstream in(model_path);

  if (!in.is_open()) {
    GTEST_SKIP() << "model.json not found at " << model_path;
  }

  json j = json::parse(in);
  in.close();

  ASSERT_NO_THROW({
    auto model = model_from_json(j);
    ASSERT_NE(model, nullptr);
  });
}
