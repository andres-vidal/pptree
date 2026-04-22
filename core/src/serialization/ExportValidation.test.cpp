#include <gtest/gtest.h>

#include "serialization/ExportValidation.hpp"

#include <filesystem>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

using json = nlohmann::json;
using namespace ppforest2::serialization;

#ifndef PPFOREST2_GOLDEN_DIR
#error "PPFOREST2_GOLDEN_DIR must be defined"
#endif

namespace {
  testing::AssertionResult throws_with(std::function<void()> const& fn, std::string const& needle) {
    try {
      fn();
    } catch (std::runtime_error const& e) {
      std::string const what = e.what();
      if (what.find(needle) != std::string::npos)
        return testing::AssertionSuccess();
      return testing::AssertionFailure() << "expected message containing \"" << needle << "\", got \"" << what << "\"";
    } catch (std::exception const& e) {
      return testing::AssertionFailure() << "wrong exception: " << e.what();
    }
    return testing::AssertionFailure() << "did not throw";
  }

  // A minimal, well-formed classification export skeleton. Enough to pass
  // validation — individual tests mutate one field to exercise rejections.
  json valid_classification_export() {
    return {
        {"model_type", "tree"},
        {"config",
         {{"mode", "classification"},
          {"size", 0},
          {"seed", 0},
          {"threads", 1},
          {"max_retries", 3},
          {"pp", {{"name", "pda"}, {"lambda", 0.0}}},
          {"vars", {{"name", "all"}}},
          {"cutpoint", {{"name", "mean_of_means"}}},
          {"stop", {{"name", "pure_node"}}},
          {"binarize", {{"name", "largest_gap"}}},
          {"grouping", {{"name", "by_label"}}},
          {"leaf", {{"name", "majority_vote"}}}}},
        {"meta",
         {{"observations", 30},
          {"features", 4},
          {"mode", "classification"},
          {"feature_names", {"a", "b", "c", "d"}},
          {"groups", {"x", "y"}}}},
        {"model", {{"root", {{"value", "x"}, {"degenerate", false}}}}}
    };
  }

  json valid_regression_export() {
    return {
        {"model_type", "forest"},
        {"config",
         {{"mode", "regression"},
          {"size", 5},
          {"seed", 0},
          {"threads", 1},
          {"max_retries", 3},
          {"pp", {{"name", "pda"}, {"lambda", 0.0}}},
          {"vars", {{"name", "all"}}},
          {"cutpoint", {{"name", "mean_of_means"}}},
          {"stop", {{"name", "min_size"}, {"min_size", 5}}},
          {"binarize", {{"name", "disabled"}}},
          {"grouping", {{"name", "by_cutpoint"}}},
          {"leaf", {{"name", "mean_response"}}}}},
        {"meta", {{"observations", 100}, {"features", 3}, {"mode", "regression"}, {"feature_names", json::array()}}},
        {"model", {{"trees", json::array()}}}
    };
  }
}

// -----------------------------------------------------------------------
// Happy paths
// -----------------------------------------------------------------------

TEST(ExportValidation, AcceptsValidClassification) {
  EXPECT_NO_THROW(validate_tree_export(valid_classification_export()));
  EXPECT_NO_THROW(validate_model_export(valid_classification_export()));
}

TEST(ExportValidation, AcceptsValidRegression) {
  EXPECT_NO_THROW(validate_forest_export(valid_regression_export()));
  EXPECT_NO_THROW(validate_model_export(valid_regression_export()));
}

// -----------------------------------------------------------------------
// Top-level skeleton
// -----------------------------------------------------------------------

TEST(ExportValidation, RejectsMissingConfig) {
  json j = valid_classification_export();
  j.erase("config");
  EXPECT_TRUE(throws_with([&] { validate_model_export(j); }, "Export.config: missing"));
}

TEST(ExportValidation, RejectsMissingMeta) {
  json j = valid_classification_export();
  j.erase("meta");
  EXPECT_TRUE(throws_with([&] { validate_model_export(j); }, "Export.meta: missing"));
}

TEST(ExportValidation, RejectsMissingModel) {
  json j = valid_classification_export();
  j.erase("model");
  EXPECT_TRUE(throws_with([&] { validate_model_export(j); }, "Export.model: missing"));
}

TEST(ExportValidation, AcceptsUnknownTopLevelKey) {
  // Top level is deliberately extensible — downstream tools (CLI, golden
  // generator) add their own annotations. Only required keys and known
  // fields are validated; extras pass through.
  json j        = valid_classification_export();
  j["surprise"] = 1;
  EXPECT_NO_THROW(validate_model_export(j));
}

TEST(ExportValidation, RejectsBogusModelType) {
  json j          = valid_classification_export();
  j["model_type"] = "random_forest";
  EXPECT_TRUE(throws_with([&] { validate_model_export(j); }, "Export.model_type"));
}

// -----------------------------------------------------------------------
// config block
// -----------------------------------------------------------------------

TEST(ExportValidation, RejectsBogusMode) {
  json j              = valid_classification_export();
  j["config"]["mode"] = "regresion";
  EXPECT_TRUE(throws_with([&] { validate_model_export(j); }, "Export.config.mode: must be one of"));
}

TEST(ExportValidation, RejectsNegativeSize) {
  json j              = valid_classification_export();
  j["config"]["size"] = -1;
  EXPECT_TRUE(throws_with([&] { validate_model_export(j); }, "Export.config.size"));
}

TEST(ExportValidation, RejectsMissingBinarizeForClassification) {
  json j = valid_classification_export();
  j["config"].erase("binarize");
  EXPECT_TRUE(throws_with([&] { validate_model_export(j); }, "Export.config.binarize: missing"));
}

TEST(ExportValidation, RejectsMissingBinarizeForRegression) {
  // Regression specs now carry `binarize::Disabled` (a mode-agnostic
  // placeholder) rather than omitting the key. Validation requires
  // `binarize` to be present for every mode.
  json j = valid_regression_export();
  j["config"].erase("binarize");
  EXPECT_TRUE(throws_with([&] { validate_model_export(j); }, "Export.config.binarize: missing"));
}

// -----------------------------------------------------------------------
// meta block and mode cross-check
// -----------------------------------------------------------------------

TEST(ExportValidation, RejectsModeMismatch) {
  json j            = valid_classification_export();
  j["meta"]["mode"] = "regression";
  EXPECT_TRUE(throws_with([&] { validate_model_export(j); }, "Export.meta.mode: disagrees with config.mode"));
}

TEST(ExportValidation, RejectsRegressionWithGroups) {
  json j              = valid_regression_export();
  j["meta"]["groups"] = {"a", "b"};
  EXPECT_TRUE(throws_with([&] { validate_model_export(j); }, "Export.meta.groups: must be absent for regression"));
}

TEST(ExportValidation, RejectsClassificationWithoutGroups) {
  json j = valid_classification_export();
  j["meta"].erase("groups");
  EXPECT_TRUE(throws_with([&] { validate_model_export(j); }, "Export.meta.groups: missing"));
}

TEST(ExportValidation, RejectsClassificationWithEmptyGroups) {
  json j              = valid_classification_export();
  j["meta"]["groups"] = json::array();
  EXPECT_TRUE(throws_with([&] { validate_model_export(j); }, "Export.meta.groups: must be non-empty"));
}

TEST(ExportValidation, RejectsNonStringGroupEntry) {
  json j              = valid_classification_export();
  j["meta"]["groups"] = json::array({"x", 42});
  EXPECT_TRUE(throws_with([&] { validate_model_export(j); }, "Export.meta.groups[1]"));
}

TEST(ExportValidation, RejectsNegativeObservations) {
  json j                    = valid_classification_export();
  j["meta"]["observations"] = -3;
  EXPECT_TRUE(throws_with([&] { validate_model_export(j); }, "Export.meta.observations"));
}

// -----------------------------------------------------------------------
// Variant-specific asserts
// -----------------------------------------------------------------------

TEST(ExportValidation, TreeExportRejectsForestModelType) {
  json j          = valid_classification_export();
  j["model_type"] = "forest";
  EXPECT_TRUE(throws_with([&] { validate_tree_export(j); }, "expected 'tree'"));
}

TEST(ExportValidation, ForestExportRejectsTreeModelType) {
  json j          = valid_regression_export();
  j["model_type"] = "tree";
  EXPECT_TRUE(throws_with([&] { validate_forest_export(j); }, "expected 'forest'"));
}

TEST(ExportValidation, TreeExportRejectsMissingRoot) {
  json j     = valid_classification_export();
  j["model"] = json::object();
  EXPECT_TRUE(throws_with([&] { validate_tree_export(j); }, "Export.model.root: missing"));
}

TEST(ExportValidation, ForestExportRejectsNonArrayTrees) {
  json j              = valid_regression_export();
  j["model"]["trees"] = 7;
  EXPECT_TRUE(throws_with([&] { validate_forest_export(j); }, "Export.model.trees"));
}

// -----------------------------------------------------------------------
// Drift protection: every committed golden file must validate.
// -----------------------------------------------------------------------

TEST(ExportValidation, AllGoldenFilesValidate) {
  std::filesystem::path const golden_dir = PPFOREST2_GOLDEN_DIR;
  ASSERT_TRUE(std::filesystem::is_directory(golden_dir)) << golden_dir;

  int checked = 0;
  for (auto const& entry : std::filesystem::recursive_directory_iterator(golden_dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".json")
      continue;

    std::ifstream in(entry.path());
    ASSERT_TRUE(in.is_open()) << entry.path();

    json j;
    in >> j;

    // Goldens don't include `model_type` at the top level (it's wrapped
    // inside the model block by the JsonModelVisitor). `validate_model_export`
    // is the variant-agnostic check that matches what goldens store.
    EXPECT_NO_THROW(validate_model_export(j)) << "golden file failed validation: " << entry.path();
    ++checked;
  }

  ASSERT_GT(checked, 0) << "no golden files found under " << golden_dir;
}
