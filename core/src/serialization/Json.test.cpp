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

#include <fstream>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using namespace ppforest2::serialization;

using json = nlohmann::json;

#ifndef PPFOREST2_GOLDEN_DIR
#error "PPFOREST2_GOLDEN_DIR must be defined"
#endif

#ifndef PPFOREST2_FIXTURES_DIR
#error "PPFOREST2_FIXTURES_DIR must be defined"
#endif

static const std::string GOLDEN_DIR   = PPFOREST2_GOLDEN_DIR;
static const std::string FIXTURES_DIR = PPFOREST2_FIXTURES_DIR;

static json load_golden(const std::string& path) {
  std::ifstream in(path);
  invariant(in.is_open(), "Required golden file missing: " + path);
  return json::parse(in);
}

TEST(JsonRoundTrip, Tree) {
  auto golden      = load_golden(GOLDEN_DIR + "/iris/tree-pda-s42.json");
  auto group_names = golden["meta"]["groups"].get<GroupNames>();
  json model_json  = golden["model"];

  Tree tree         = tree_from_json(model_json, group_names);
  json roundtripped = to_json(tree, group_names);

  ASSERT_EQ(model_json, roundtripped) << "Tree JSON should be identical after round-trip";
}

TEST(JsonRoundTrip, Forest) {
  auto golden      = load_golden(GOLDEN_DIR + "/iris/forest-pda-t5-s42.json");
  auto group_names = golden["meta"]["groups"].get<GroupNames>();
  json model_json  = golden["model"];

  Forest forest     = forest_from_json(model_json, group_names);
  json roundtripped = to_json(forest, group_names);

  ASSERT_EQ(model_json, roundtripped) << "Forest JSON should be identical after round-trip";
}

TEST(JsonRoundTrip, ModelDispatchTree) {
  auto golden      = load_golden(GOLDEN_DIR + "/iris/tree-pda-s42.json");
  auto group_names = golden["meta"]["groups"].get<GroupNames>();
  json model_json  = golden["model"];

  json wrapped;
  wrapped["model_type"] = "tree";
  wrapped["model"]      = model_json;

  // model_from_json uses integer format — test with integer-format JSON
  json int_model = to_json(tree_from_json(model_json, group_names));
  json int_wrapped;
  int_wrapped["model_type"] = "tree";
  int_wrapped["model"]      = int_model;

  auto restored = model_from_json(int_wrapped);
  ASSERT_NE(restored, nullptr);

  json roundtripped = to_json(*restored);
  ASSERT_EQ(roundtripped["model_type"], "tree");
}

TEST(JsonRoundTrip, ModelDispatchForest) {
  auto golden      = load_golden(GOLDEN_DIR + "/iris/forest-pda-t5-s42.json");
  auto group_names = golden["meta"]["groups"].get<GroupNames>();
  json model_json  = golden["model"];

  // model_from_json uses integer format — test with integer-format JSON
  json int_model = to_json(forest_from_json(model_json, group_names));
  json int_wrapped;
  int_wrapped["model_type"] = "forest";
  int_wrapped["model"]      = int_model;

  auto restored = model_from_json(int_wrapped);
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
  ASSERT_TRUE(j.contains("group_errors"));

  auto matrix = j["matrix"].get<std::vector<std::vector<int>>>();
  ASSERT_EQ(matrix.size(), 3);
  ASSERT_EQ(matrix[0].size(), 3);
}

// ---------------------------------------------------------------------------
// Labeled serialization tests
// ---------------------------------------------------------------------------

TEST(JsonLabeled, TreeLeafValuesAreStrings) {
  auto golden      = load_golden(GOLDEN_DIR + "/iris/tree-pda-s42.json");
  auto group_names = golden["meta"]["groups"].get<GroupNames>();

  Tree tree = tree_from_json(golden["model"], group_names);
  json j    = to_json(tree, group_names);

  // Walk to a leaf
  const json *node = &j["root"];

  while (node->contains("lower"))
    node = &(*node)["lower"];

  ASSERT_TRUE((*node)["value"].is_string());
  auto value = (*node)["value"].get<std::string>();
  EXPECT_TRUE(std::find(group_names.begin(), group_names.end(), value) != group_names.end()) << "Leaf value '" << value << "' not in group_names";
}

TEST(JsonLabeled, TreeGroupsAreStrings) {
  auto golden      = load_golden(GOLDEN_DIR + "/iris/tree-pda-s42.json");
  auto group_names = golden["meta"]["groups"].get<GroupNames>();

  Tree tree = tree_from_json(golden["model"], group_names);
  json j    = to_json(tree, group_names);

  auto root_groups = j["root"]["groups"].get<std::vector<std::string>>();
  EXPECT_EQ(root_groups.size(), group_names.size());

  for (const auto& g : root_groups) {
    EXPECT_TRUE(std::find(group_names.begin(), group_names.end(), g) != group_names.end()) << "Group '" << g << "' not in group_names";
  }
}

TEST(JsonLabeled, IntegerFormatOmitsGroupNames) {
  auto golden      = load_golden(GOLDEN_DIR + "/iris/tree-pda-s42.json");
  auto group_names = golden["meta"]["groups"].get<GroupNames>();

  Tree tree = tree_from_json(golden["model"], group_names);
  json j    = to_json(tree);

  // Walk to a leaf
  const json *node = &j["root"];

  while (node->contains("lower"))
    node = &(*node)["lower"];

  ASSERT_TRUE((*node)["value"].is_number());
}

TEST(JsonLabeled, ConfusionMatrixLabelsAreStrings) {
  ResponseVector predictions(6);
  predictions << 0, 0, 1, 1, 2, 2;

  ResponseVector actual(6);
  actual << 0, 1, 1, 1, 2, 0;

  GroupNames names = { "setosa", "versicolor", "virginica" };
  ConfusionMatrix cm(predictions, actual);
  json j = to_json(cm, names);

  auto labels = j["labels"].get<std::vector<std::string>>();
  EXPECT_EQ(labels, names);
}

TEST(JsonLabeled, ForestRoundTrip) {
  auto golden      = load_golden(GOLDEN_DIR + "/iris/forest-pda-t5-s42.json");
  auto group_names = golden["meta"]["groups"].get<GroupNames>();
  json model_json  = golden["model"];

  Forest forest     = forest_from_json(model_json, group_names);
  json roundtripped = to_json(forest, group_names);

  ASSERT_EQ(model_json, roundtripped) << "Forest labeled round-trip should be identical";
}

TEST(JsonLabeled, LabeledAndIntegerProduceSameModel) {
  auto golden      = load_golden(GOLDEN_DIR + "/iris/tree-pda-s42.json");
  auto group_names = golden["meta"]["groups"].get<GroupNames>();

  Tree from_labeled = tree_from_json(golden["model"], group_names);

  // Convert to integer format and back
  json int_json     = to_json(from_labeled);
  Tree from_integer = tree_from_json(int_json);

  // Both should produce the same integer-format JSON
  EXPECT_EQ(to_json(from_labeled), to_json(from_integer));
}

TEST(JsonRoundTrip, ModelFromJsonFile) {
  std::string model_path = FIXTURES_DIR + "/model.json";
  std::ifstream in(model_path);

  invariant(in.is_open(), "model.json not found at " + model_path);

  json j = json::parse(in);
  in.close();

  ASSERT_NO_THROW({
    auto model = model_from_json(j);
    ASSERT_NE(model, nullptr);
  });
}
