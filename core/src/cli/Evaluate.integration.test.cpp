/**
 * @file Evaluate.integration.test.cpp
 * @brief Integration tests for the evaluate subcommand.
 */
#include "cli/CLI.integration.hpp"

// ---------------------------------------------------------------------------
// Evaluate subcommand
// ---------------------------------------------------------------------------

/* Basic evaluation with simulated data succeeds. */
TEST(CLIEvaluate, EvaluateWithSimulatedData) {
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 1");
  EXPECT_EQ(result.exit_code, 0);
}

/* Evaluation with real iris data succeeds. */
TEST(CLIEvaluate, EvaluateWithIrisData) {
  auto result = run_ppforest2("-q evaluate -d " + IRIS_CSV + " -n 5 -r 0 -i 1");
  EXPECT_EQ(result.exit_code, 0);
}

/* Non-quiet evaluate prints results header and error metrics. */
TEST(CLIEvaluate, EvaluateTextOutput) {
  auto result = run_ppforest2("evaluate --simulate 50x3x2 -n 5 -r 0 -i 1");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("Evaluation results"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Train Err"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Test Err"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Peak RSS"), std::string::npos);
}

/* Evaluation with a single tree (t=0) succeeds. */
TEST(CLIEvaluate, EvaluateSingleTree) {
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 0 -r 0 -i 1");
  EXPECT_EQ(result.exit_code, 0);
}

/* -o writes evaluation stats (runs, mean_time, peak_rss) to JSON. */
TEST(CLIEvaluate, EvaluateOutputFile) {
  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 1 -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("runs"));
  EXPECT_TRUE(j.contains("mean_time_ms"));
  EXPECT_TRUE(j.contains("peak_rss_bytes"));
}

/* -e exports config.json, data.csv, and results.json to a directory. */
TEST(CLIEvaluate, EvaluateExport) {
  TempDir dir;
  std::string export_path = dir.path() + "/exp1";

  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -l 0.3 -i 1 -e " + export_path);
  EXPECT_EQ(result.exit_code, 0);

  EXPECT_TRUE(std::filesystem::exists(export_path + "/config.json"));
  EXPECT_TRUE(std::filesystem::exists(export_path + "/data.csv"));
  EXPECT_TRUE(std::filesystem::exists(export_path + "/results.json"));

  // Verify config.json captures all configuration for reproducibility
  std::ifstream config_in(export_path + "/config.json");
  auto config = json::parse(config_in);
  EXPECT_EQ(config["data"], "data.csv");
  EXPECT_EQ(config["size"], 5);
  EXPECT_FLOAT_EQ(config["pp"]["lambda"].get<float>(), 0.3f);
  EXPECT_EQ(config["seed"], 0);
  EXPECT_TRUE(config.contains("threads"));
  EXPECT_TRUE(config.contains("train-ratio"));
  EXPECT_TRUE(config.contains("iterations"));
}

/* Output to an existing file must fail. */
TEST(CLIEvaluate, EvaluateOutputCollisionFails) {
  TempFile output;
  // Don't clear - file exists, should fail
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 1 -o " + output.path());
  EXPECT_NE(result.exit_code, 0);
}

/* Export to an existing directory must fail. */
TEST(CLIEvaluate, EvaluateExportCollisionFails) {
  TempDir dir;
  // dir.path() already exists, should fail
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 1 -e " + dir.path());
  EXPECT_NE(result.exit_code, 0);
}

/* JSON output includes an iterations array with per-run metrics. */
TEST(CLIEvaluate, EvaluateOutputHasIterationsArray) {
  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 1 -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("iterations"));
  EXPECT_EQ(j["iterations"].size(), 1u);
  EXPECT_TRUE(j["iterations"][0].contains("train_time_ms"));
  EXPECT_TRUE(j["iterations"][0].contains("train_error"));
  EXPECT_TRUE(j["iterations"][0].contains("test_error"));
}

/* Multiple iterations (-i 3) produce matching array size. */
TEST(CLIEvaluate, EvaluateMultipleIterationsArray) {
  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 3 -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("iterations"));
  EXPECT_EQ(j["iterations"].size(), 3u);
  EXPECT_EQ(j["runs"], 3);

  for (const auto& iter : j["iterations"]) {
    EXPECT_TRUE(iter.contains("train_time_ms"));
    EXPECT_TRUE(iter.contains("train_error"));
    EXPECT_TRUE(iter.contains("test_error"));
    EXPECT_FALSE(iter.contains("peak_rss"));
  }
}

/* Evaluate without -d or --simulate must fail. */
TEST(CLIEvaluate, EvaluateNoDataSourceFails) {
  auto result = run_ppforest2("evaluate");
  EXPECT_NE(result.exit_code, 0);
}

/* Malformed --simulate string must fail. */
TEST(CLIEvaluate, EvaluateInvalidSimFormatFails) {
  auto result = run_ppforest2("evaluate --simulate 100x5");
  EXPECT_NE(result.exit_code, 0);
}

/* Fraction "1/3" is accepted end-to-end for evaluate. */
TEST(CLIEvaluate, EvaluateWithFractionVars) {
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 1 -v 1/3");
  EXPECT_EQ(result.exit_code, 0);
}
