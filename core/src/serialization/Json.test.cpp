/**
 * @file Json.test.cpp
 * @brief Tests for JSON serialization round-trips.
 *
 * Uses committed golden files as fixtures — loads model JSON, deserializes,
 * re-serializes, and compares.  No training is performed.
 */
#include <gtest/gtest.h>

#include "serialization/Json.hpp"
#include "models/Bagged.hpp"
#include "models/ClassificationTree.hpp"
#include "models/ClassificationForest.hpp"
#include "models/RegressionTree.hpp"
#include "models/RegressionForest.hpp"
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
static std::string const FIXTURES_DIR = PPFOREST2_FIXTURES_DIR;

static json load_golden(std::string const& path) {
  std::ifstream in(path);
  invariant(in.is_open(), "Required golden file missing: " + path);
  return json::parse(in);
}

TEST(JsonRoundTrip, Tree) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Tree::Ptr>>();

  json roundtripped = to_json(*e.model, e.groups);

  ASSERT_EQ(golden["model"], roundtripped) << "Tree JSON should be identical after round-trip";
}

TEST(JsonRoundTrip, Forest) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");
  auto e      = golden.get<Export<Forest::Ptr>>();

  json roundtripped = to_json(*e.model, e.groups);

  ASSERT_EQ(golden["model"], roundtripped) << "Forest JSON should be identical after round-trip";
}

TEST(JsonRoundTrip, ModelDispatchTree) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");

  auto e = golden.get<Export<Model::Ptr>>();
  ASSERT_NE(e.model, nullptr);

  json roundtripped = to_json(*e.model);
  ASSERT_EQ(roundtripped["model_type"], "tree");
}

TEST(JsonRoundTrip, ModelDispatchForest) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");

  auto e = golden.get<Export<Model::Ptr>>();
  ASSERT_NE(e.model, nullptr);

  json roundtripped = to_json(*e.model);
  ASSERT_EQ(roundtripped["model_type"], "forest");
}

TEST(JsonRoundTrip, ConfusionMatrix) {
  GroupIdVector predictions(6);
  predictions << 0, 0, 1, 1, 2, 2;

  GroupIdVector actual(6);
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
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Tree::Ptr>>();
  json j      = to_json(*e.model, e.groups);

  // Walk to a leaf
  json const* node = &j["root"];

  while (node->contains("lower"))
    node = &(*node)["lower"];

  ASSERT_TRUE((*node)["value"].is_string());
  auto value = (*node)["value"].get<std::string>();
  EXPECT_TRUE(std::find(e.groups.begin(), e.groups.end(), value) != e.groups.end())
      << "Leaf value '" << value << "' not in group_names";
}

TEST(JsonLabeled, TreeGroupsAreStrings) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Tree::Ptr>>();
  json j      = to_json(*e.model, e.groups);

  auto root_groups = j["root"]["groups"].get<std::vector<std::string>>();
  EXPECT_EQ(root_groups.size(), e.groups.size());

  for (auto const& g : root_groups) {
    EXPECT_TRUE(std::find(e.groups.begin(), e.groups.end(), g) != e.groups.end())
        << "Group '" << g << "' not in group_names";
  }
}

TEST(JsonLabeled, IntegerFormatOmitsGroupNames) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Tree::Ptr>>();
  json j      = to_json(*e.model);

  // Walk to a leaf
  json const* node = &j["root"];

  while (node->contains("lower"))
    node = &(*node)["lower"];

  ASSERT_TRUE((*node)["value"].is_number());
}

TEST(JsonLabeled, ConfusionMatrixLabelsAreStrings) {
  GroupIdVector predictions(6);
  predictions << 0, 0, 1, 1, 2, 2;

  GroupIdVector actual(6);
  actual << 0, 1, 1, 1, 2, 0;

  GroupNames names = {"setosa", "versicolor", "virginica"};
  ConfusionMatrix cm(predictions, actual);
  json j = to_json(cm, names);

  auto labels = j["labels"].get<std::vector<std::string>>();
  EXPECT_EQ(labels, names);
}

TEST(JsonLabeled, ForestRoundTrip) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");
  auto e      = golden.get<Export<Forest::Ptr>>();

  json roundtripped = to_json(*e.model, e.groups);

  ASSERT_EQ(golden["model"], roundtripped) << "Forest labeled round-trip should be identical";
}

TEST(JsonLabeled, LabeledAndIntegerProduceSameModel) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Tree::Ptr>>();

  // Convert to integer format and back
  json int_json     = to_json(*e.model);
  auto from_integer = int_json.get<Tree::Ptr>();

  // Both should produce the same integer-format JSON
  EXPECT_EQ(to_json(*e.model), to_json(*from_integer));
}

TEST(JsonRoundTrip, ModelFromJsonFile) {
  std::string model_path = FIXTURES_DIR + "/model.json";
  std::ifstream in(model_path);

  invariant(in.is_open(), "model.json not found at " + model_path);

  json j = json::parse(in);
  in.close();

  ASSERT_NO_THROW({
    auto e = j.get<Export<Model::Ptr>>();
    ASSERT_NE(e.model, nullptr);
  });
}

// ---------------------------------------------------------------------------
// Serialization structure tests — verify JSON shape without training
// ---------------------------------------------------------------------------

TEST(JsonStructure, TreeAlwaysHasDegenerate) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Tree::Ptr>>();
  json j      = to_json(*e.model, e.groups);

  EXPECT_TRUE(j.contains("degenerate"));
  EXPECT_FALSE(j["degenerate"].get<bool>());
}

TEST(JsonStructure, ForestAlwaysHasDegenerate) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");
  auto e      = golden.get<Export<Forest::Ptr>>();
  json j      = to_json(*e.model, e.groups);

  EXPECT_TRUE(j.contains("degenerate"));
  EXPECT_FALSE(j["degenerate"].get<bool>());
}

TEST(JsonStructure, ConfusionMatrixRoundTrip) {
  GroupIdVector predictions(6);
  predictions << 0, 0, 1, 1, 2, 2;

  GroupIdVector actual(6);
  actual << 0, 1, 1, 1, 2, 0;

  ConfusionMatrix cm(predictions, actual);
  json j = to_json(cm);

  auto restored = j.get<ConfusionMatrix>();

  EXPECT_EQ(restored.values.rows(), cm.values.rows());
  EXPECT_EQ(restored.values.cols(), cm.values.cols());
  EXPECT_EQ(restored.values, cm.values);
}

TEST(JsonStructure, ForestSampleIndicesRoundTrip) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");
  auto e      = golden.get<Export<Forest::Ptr>>();

  json roundtripped = to_json(*e.model, e.groups);

  for (size_t i = 0; i < e.model->trees.size(); ++i) {
    auto* bt = dynamic_cast<BaggedTree const*>(e.model->trees[i].get());
    ASSERT_NE(bt, nullptr) << "Tree " << i << " should be a BaggedTree";
    EXPECT_FALSE(bt->sample_indices.empty()) << "Tree " << i << " should have sample_indices";

    auto rt_indices = roundtripped["trees"][i]["sample_indices"].get<std::vector<int>>();
    EXPECT_EQ(rt_indices, bt->sample_indices) << "Tree " << i << " sample_indices should round-trip";
  }
}

TEST(JsonStructure, VariableImportanceRoundTrip) {
  VariableImportance vi;
  vi.scale       = types::FeatureVector::Ones(4);
  vi.projections = types::FeatureVector::Random(4);

  json j = to_json(vi);

  auto restored = j.get<VariableImportance>();

  EXPECT_EQ(restored.projections.size(), vi.projections.size());
  EXPECT_EQ(restored.scale.size(), vi.scale.size());
}

// ---------------------------------------------------------------------------
// Export round-trip tests — full JSON (model + config + meta + metrics)
// ---------------------------------------------------------------------------

TEST(ExportRoundTrip, TreePreservesModelAndMeta) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Model::Ptr>>();

  json roundtripped = e.to_json();

  EXPECT_TRUE(roundtripped.contains("model_type"));
  EXPECT_EQ(roundtripped["model_type"], "tree");
  EXPECT_TRUE(roundtripped.contains("config"));
  EXPECT_TRUE(roundtripped.contains("meta"));
  EXPECT_EQ(roundtripped["meta"]["groups"], golden["meta"]["groups"]);
  EXPECT_EQ(roundtripped["meta"]["observations"], golden["meta"]["observations"]);
  EXPECT_EQ(roundtripped["meta"]["features"], golden["meta"]["features"]);
  EXPECT_EQ(roundtripped["meta"]["feature_names"], golden["meta"]["feature_names"]);
  EXPECT_EQ(roundtripped["model"], golden["model"]);
}

TEST(ExportRoundTrip, ForestPreservesModelAndMeta) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");
  auto e      = golden.get<Export<Model::Ptr>>();

  json roundtripped = e.to_json();

  EXPECT_TRUE(roundtripped.contains("model_type"));
  EXPECT_EQ(roundtripped["model_type"], "forest");
  EXPECT_TRUE(roundtripped.contains("config"));
  EXPECT_TRUE(roundtripped.contains("meta"));
  EXPECT_EQ(roundtripped["meta"]["groups"], golden["meta"]["groups"]);
  EXPECT_EQ(roundtripped["model"], golden["model"]);
}

TEST(ExportRoundTrip, TreePreservesConfig) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Model::Ptr>>();

  json roundtripped = e.to_json();

  EXPECT_EQ(roundtripped["config"]["pp"]["method"], golden["config"]["pp"]["method"]);
  EXPECT_EQ(roundtripped["config"]["pp"]["lambda"], golden["config"]["pp"]["lambda"]);
}

TEST(ExportRoundTrip, TreePreservesMetrics) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Model::Ptr>>();

  json roundtripped = e.to_json();

  EXPECT_TRUE(roundtripped.contains("training_confusion_matrix"));
  EXPECT_EQ(roundtripped["training_confusion_matrix"], golden["training_confusion_matrix"]);
  EXPECT_TRUE(roundtripped.contains("variable_importance"));
  EXPECT_EQ(roundtripped["variable_importance"], golden["variable_importance"]);
}

TEST(ExportRoundTrip, ForestPreservesMetrics) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");
  auto e      = golden.get<Export<Model::Ptr>>();

  json roundtripped = e.to_json();

  EXPECT_TRUE(roundtripped.contains("training_confusion_matrix"));
  EXPECT_EQ(roundtripped["training_confusion_matrix"], golden["training_confusion_matrix"]);
  EXPECT_TRUE(roundtripped.contains("variable_importance"));
  EXPECT_EQ(roundtripped["variable_importance"], golden["variable_importance"]);
  EXPECT_TRUE(roundtripped.contains("oob_error"));
  EXPECT_DOUBLE_EQ(roundtripped["oob_error"].get<double>(), golden["oob_error"].get<double>());
  EXPECT_TRUE(roundtripped.contains("oob_confusion_matrix"));
  EXPECT_EQ(roundtripped["oob_confusion_matrix"], golden["oob_confusion_matrix"]);
}

TEST(ExportRoundTrip, FullRoundTripTreeIdentical) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Model::Ptr>>();

  json roundtripped = e.to_json();

  // Re-deserialize and re-serialize — should be stable
  auto e2    = roundtripped.get<Export<Model::Ptr>>();
  json again = e2.to_json();

  EXPECT_EQ(roundtripped["model"], again["model"]);
  EXPECT_EQ(roundtripped["config"], again["config"]);
  EXPECT_EQ(roundtripped["meta"], again["meta"]);
}

TEST(ExportRoundTrip, FullRoundTripForestIdentical) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");
  auto e      = golden.get<Export<Model::Ptr>>();

  json roundtripped = e.to_json();

  auto e2    = roundtripped.get<Export<Model::Ptr>>();
  json again = e2.to_json();

  EXPECT_EQ(roundtripped["model"], again["model"]);
  EXPECT_EQ(roundtripped["config"], again["config"]);
  EXPECT_EQ(roundtripped["meta"], again["meta"]);
}
