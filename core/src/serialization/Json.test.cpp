/**
 * @file Json.test.cpp
 * @brief Tests for JSON serialization round-trips.
 *
 * Uses committed golden files as fixtures — loads model JSON, deserializes,
 * re-serializes, and compares.  No training is performed.
 */
#include <gtest/gtest.h>

#include "serialization/Json.hpp"
#include "utils/Invariant.hpp"
#include "utils/Macros.hpp"

#include <filesystem>
#include <fstream>

using namespace pptree;
using namespace pptree::types;
using namespace pptree::stats;
using namespace pptree::serialization;

using json = nlohmann::json;

#ifndef PPTREE_GOLDEN_DIR
#error "PPTREE_GOLDEN_DIR must be defined"
#endif

#ifndef PPTREE_DATA_DIR
#error "PPTREE_DATA_DIR must be defined"
#endif

static const std::string GOLDEN_DIR = PPTREE_GOLDEN_DIR;
static const std::string DATA_DIR   = PPTREE_DATA_DIR;

static json load_model_json(const std::string& path) {
  std::ifstream in(path);
  invariant(in.is_open(), "Required golden file missing: " + path);
  json golden = json::parse(in);
  return golden["model"];
}

TEST(JsonRoundTrip, Tree) {
  json model_json = load_model_json(GOLDEN_DIR + "/iris/tree-glda-s42.json");

  Tree tree = tree_from_json(model_json);
  json roundtripped = to_json(tree);

  ASSERT_EQ(model_json, roundtripped) << "Tree JSON should be identical after round-trip";
}

TEST(JsonRoundTrip, Forest) {
  json model_json = load_model_json(GOLDEN_DIR + "/iris/forest-glda-t5-s42.json");

  Forest forest = forest_from_json(model_json);
  json roundtripped = to_json(forest);

  ASSERT_EQ(model_json, roundtripped) << "Forest JSON should be identical after round-trip";
}

TEST(JsonRoundTrip, ModelDispatchTree) {
  json model_json = load_model_json(GOLDEN_DIR + "/iris/tree-glda-s42.json");

  json wrapped;
  wrapped["model_type"] = "tree";
  wrapped["model"] = model_json;

  auto restored = model_from_json(wrapped);
  ASSERT_NE(restored, nullptr);

  json roundtripped = to_json(*restored);
  ASSERT_EQ(roundtripped["model_type"], "tree");
}

TEST(JsonRoundTrip, ModelDispatchForest) {
  json model_json = load_model_json(GOLDEN_DIR + "/iris/forest-glda-t5-s42.json");

  json wrapped;
  wrapped["model_type"] = "forest";
  wrapped["model"] = model_json;

  auto restored = model_from_json(wrapped);
  ASSERT_NE(restored, nullptr);

  json roundtripped = to_json(*restored);
  ASSERT_EQ(roundtripped["model_type"], "forest");
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

TEST(JsonRoundTrip, FromModelJsonFile) {
  std::string model_path = DATA_DIR + "/../model.json";
  std::ifstream in(model_path);

  invariant(in.is_open(), "model.json not found at " + model_path);

  json j = json::parse(in);
  in.close();

  ASSERT_NO_THROW({
    auto model = model_from_json(j);
    ASSERT_NE(model, nullptr);
  });
}
