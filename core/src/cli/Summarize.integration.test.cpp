/**
 * @file Summarize.integration.test.cpp
 * @brief Integration tests for the summarize subcommand and saved model structure.
 */
#include "cli/CLI.integration.hpp"

// ---------------------------------------------------------------------------
// Summarize subcommand
// ---------------------------------------------------------------------------

class SummarizeTest : public SavedModelTest {};

/* Summarize must succeed on a saved model and exit 0. */
TEST_F(SummarizeTest, SummarizeExitCode) {
  auto result = run_ppforest2("-q summarize -M " + model_->path());
  EXPECT_EQ(result.exit_code, 0);
}

/* Summarize with a nonexistent model file must fail. */
TEST(CLISummarize, SummarizeMissingModelFails) {
  auto result = run_ppforest2("-q summarize -M /nonexistent/model.json");
  EXPECT_NE(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Saved model JSON structure
// ---------------------------------------------------------------------------

/* Saved model must contain training_confusion_matrix. */
TEST_F(SummarizeTest, HasTrainingConfusionMatrix) {
  EXPECT_TRUE(model_json_.contains("training_confusion_matrix"));
  EXPECT_TRUE(model_json_["training_confusion_matrix"].contains("matrix"));
  EXPECT_TRUE(model_json_["training_confusion_matrix"].contains("labels"));
  EXPECT_TRUE(model_json_["training_confusion_matrix"].contains("group_errors"));
}

/* Saved model must contain oob_confusion_matrix. */
TEST_F(SummarizeTest, HasOOBConfusionMatrix) {
  EXPECT_TRUE(model_json_.contains("oob_confusion_matrix"));
  EXPECT_TRUE(model_json_["oob_confusion_matrix"].contains("matrix"));
  EXPECT_TRUE(model_json_["oob_confusion_matrix"].contains("labels"));
  EXPECT_TRUE(model_json_["oob_confusion_matrix"].contains("group_errors"));
}

/* Saved model meta must contain observations and features. */
TEST_F(SummarizeTest, MetaHasDataDimensions) {
  ASSERT_TRUE(model_json_.contains("meta"));
  EXPECT_EQ(model_json_["meta"]["observations"].get<int>(), 150);
  EXPECT_EQ(model_json_["meta"]["features"].get<int>(), 4);
  EXPECT_EQ(model_json_["meta"]["groups"].size(), 3u);
}

/* Saved model must contain training duration. */
TEST_F(SummarizeTest, HasTrainingDuration) {
  EXPECT_TRUE(model_json_.contains("training_duration_ms"));
  EXPECT_GE(model_json_["training_duration_ms"].get<long long>(), 0);
}

/* Saved model must not contain ephemeral fields. */
TEST_F(SummarizeTest, NoEphemeralFields) {
  EXPECT_FALSE(model_json_.contains("save_path"));
}

// ---------------------------------------------------------------------------
// Summarize with --no-metrics model
// ---------------------------------------------------------------------------

/* Summarize succeeds on a model trained with --no-metrics. */
TEST(CLISummarize, SummarizeNoMetricsModel) {
  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --no-metrics -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  auto result = run_ppforest2("--no-color summarize -M " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  // Config and data summary should still be shown
  EXPECT_NE(result.stdout_output.find("Random Forest"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Data Summary"), std::string::npos);

  // Metrics should not be shown
  EXPECT_EQ(result.stdout_output.find("Training Confusion Matrix"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("OOB"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Variable Importance"), std::string::npos);
}

/* Summarize with --data recomputes metrics for a --no-metrics model. */
TEST(CLISummarize, SummarizeWithDataRecomputesMetrics) {
  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --no-metrics -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  auto result = run_ppforest2("--no-color summarize -M " + model.path() + " -d " + IRIS_CSV);
  EXPECT_EQ(result.exit_code, 0);

  // Metrics should now be shown
  EXPECT_NE(result.stdout_output.find("Training Confusion Matrix"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("OOB Confusion Matrix"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Variable Importance"), std::string::npos);
}

/* Summarize with --data does not recompute when metrics already exist. */
TEST_F(SummarizeTest, SummarizeWithDataSkipsExistingMetrics) {
  // Model already has metrics — providing --data should not change output
  auto result = run_ppforest2("-q summarize -M " + model_->path() + " -d " + IRIS_CSV);
  EXPECT_EQ(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Saved model — bootstrap sample indices
// ---------------------------------------------------------------------------

/* Saved forest must contain sample_indices for each tree. */
TEST_F(SummarizeTest, HasSampleIndices) {
  auto trees = model_json_["model"]["trees"];
  ASSERT_GT(trees.size(), 0u);

  for (auto const& tree : trees) {
    ASSERT_TRUE(tree.contains("sample_indices")) << "Each tree must have sample_indices";
    EXPECT_GT(tree["sample_indices"].size(), 0u);
  }
}

// ---------------------------------------------------------------------------
// Summarize with non-default strategies
// ---------------------------------------------------------------------------

/* Summarize shows strategy display names for non-default strategies. */
TEST(CLISummarize, SummarizeNonDefaultStrategies) {
  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + IRIS_CSV +
                             " -n 5 -r 0 "
                             "--pp pda:lambda=0.5 --dr noop -s " +
                             model.path());
  ASSERT_EQ(train.exit_code, 0);

  auto result = run_ppforest2("--no-color summarize -M " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  // Strategy display names should appear in summary output
  EXPECT_NE(result.stdout_output.find("PDA"), std::string::npos) << "Expected PDA strategy display name in summary";
}
