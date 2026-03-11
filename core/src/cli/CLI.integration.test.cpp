/**
 * @file CLI.integration.test.cpp
 * @brief Integration tests for the pptree command-line interface.
 *
 * Each test spawns the pptree binary as a child process and checks its
 * exit code and stdout output.  Temporary files (TempFile, TempDir) are
 * used for model and output artifacts; they are automatically cleaned up.
 */
#include <gtest/gtest.h>

#include "io/TempFile.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>
#include <filesystem>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

#ifndef PPTREE_BINARY_PATH
#error "PPTREE_BINARY_PATH must be defined"
#endif

#ifndef PPTREE_DATA_DIR
#error "PPTREE_DATA_DIR must be defined"
#endif

#ifndef PPTREE_GOLDEN_DIR
#error "PPTREE_GOLDEN_DIR must be defined"
#endif

static const std::string BINARY     = PPTREE_BINARY_PATH;
static const std::string DATA_DIR   = PPTREE_DATA_DIR;
static const std::string GOLDEN_DIR = PPTREE_GOLDEN_DIR;
static const std::string IRIS_CSV   = DATA_DIR + "/iris.csv";
static const std::string CRAB_CSV   = DATA_DIR + "/crab.csv";
static const std::string WINE_CSV   = DATA_DIR + "/wine.csv";
static const std::string GLASS_CSV  = DATA_DIR + "/glass.csv";

// ---------------------------------------------------------------------------
// Test utilities — process runner, temporary file/directory helpers
// ---------------------------------------------------------------------------

/**
 * @brief Captured output of a child-process invocation.
 *
 * Holds the exit code and the entire stdout text so tests can assert
 * both process success and textual output content.
 */
struct ProcessResult {
  int exit_code;
  std::string stdout_output;
};

/**
 * @brief Spawn the pptree binary with the given argument string.
 *
 * Stderr is redirected to /dev/null (NUL on Windows) so only stdout
 * is captured.  The exit code is extracted via WEXITSTATUS on POSIX.
 *
 * @param args  Space-separated argument string appended to the binary path.
 * @return ProcessResult with exit code and captured stdout.
 */
static ProcessResult run_pptree(const std::string& args) {
  #ifdef _WIN32
  std::string cmd = BINARY + " " + args + " 2>NUL";
  FILE *pipe      = _popen(cmd.c_str(), "r");
  #else
  std::string cmd = BINARY + " " + args + " 2>/dev/null";
  FILE *pipe      = popen(cmd.c_str(), "r");
  #endif

  if (!pipe) {
    return { -1, "" };
  }

  std::string output;
  char buffer[4096];

  while (fgets(buffer, sizeof(buffer), pipe))
    output += buffer;

  #ifdef _WIN32
  int exit_code = _pclose(pipe);
  #else
  int status    = pclose(pipe);
  int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  #endif

  return { exit_code, output };
}

using pptree::io::TempFile;
using pptree::io::TempDir;

// ---------------------------------------------------------------------------
// Golden test helpers — load reference files and compare fields
// ---------------------------------------------------------------------------

static json load_golden(const std::string& dataset, const std::string& slug) {
  std::string path = GOLDEN_DIR + "/" + dataset + "/" + slug + ".json";
  std::ifstream in(path);
  EXPECT_TRUE(in.good()) << "Golden file not found: " << path;
  return json::parse(in);
}

static void compare_predictions(const json& actual, const json& expected) {
  ASSERT_EQ(actual.size(), expected.size()) << "predictions size mismatch";
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i].get<int>(), expected[i].get<int>()) << "prediction[" << i << "]";
  }
}

static void compare_float_array(
  const json& actual, const json& expected, double tol, const std::string& name) {
  ASSERT_EQ(actual.size(), expected.size()) << name << " size mismatch";
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual[i].get<double>(), expected[i].get<double>(), tol)
      << name << "[" << i << "]";
  }
}

static void compare_confusion_matrix(const json& actual, const json& expected) {
  EXPECT_EQ(actual["matrix"], expected["matrix"]) << "confusion matrix mismatch";
  EXPECT_EQ(actual["labels"], expected["labels"]) << "confusion labels mismatch";
}

// ---------------------------------------------------------------------------
// Golden test macros — train via CLI, predict, compare against golden files
// ---------------------------------------------------------------------------

#define CLI_GOLDEN_TREE_TEST(TestName, dataset, slug, csv, lambda, seed)                                                           \
        TEST(CLIGolden, TestName) {                                                                                                \
          auto golden = load_golden(dataset, slug);                                                                                \
                                                                                                                                   \
          TempFile model;                                                                                                          \
          model.clear();                                                                                                           \
          auto train = run_pptree(                                                                                                 \
            "-q train -d " + csv + " -t 0"                                                                                         \
            " -l " + std::to_string(lambda) +                                                                                      \
            " -r " + std::to_string(seed) +                                                                                        \
            " -s " + model.path());                                                                                                \
          ASSERT_EQ(train.exit_code, 0) << "train failed";                                                                         \
                                                                                                                                   \
          auto model_json = json::parse(model.read());                                                                             \
          ASSERT_TRUE(model_json.contains("variable_importance"));                                                                 \
          auto vi = model_json["variable_importance"];                                                                             \
          compare_float_array(vi["scale"], golden["variable_importance"]["scale"], 1e-3, #TestName " VI scale");                   \
          compare_float_array(vi["projections"], golden["variable_importance"]["projections"], 1e-3, #TestName " VI projections"); \
                                                                                                                                   \
          TempFile output;                                                                                                         \
          output.clear();                                                                                                          \
          auto predict = run_pptree("-q predict -M " + model.path() + " -d " + csv + " -o " + output.path());                      \
          ASSERT_EQ(predict.exit_code, 0) << "predict failed";                                                                     \
                                                                                                                                   \
          auto pred_json = json::parse(output.read());                                                                             \
          compare_predictions(pred_json["predictions"], golden["predictions"]);                                                    \
          EXPECT_NEAR(pred_json["error_rate"].get<double>(), golden["error_rate"].get<double>(), 1e-3) << #TestName " error_rate"; \
          compare_confusion_matrix(pred_json["confusion_matrix"], golden["confusion_matrix"]);                                     \
        }

#define CLI_GOLDEN_FOREST_TEST(TestName, dataset, slug, csv, n_trees, lambda, n_vars, seed)                                         \
        TEST(CLIGolden, TestName) {                                                                                                 \
          auto golden = load_golden(dataset, slug);                                                                                 \
                                                                                                                                    \
          TempFile model;                                                                                                           \
          model.clear();                                                                                                            \
          auto train = run_pptree(                                                                                                  \
            "-q train -d " + csv +                                                                                                  \
            " -t " + std::to_string(n_trees) +                                                                                      \
            " -l " + std::to_string(lambda) +                                                                                       \
            " -r " + std::to_string(seed) +                                                                                         \
            " -v " + std::to_string(n_vars) +                                                                                       \
            " --threads 1"                                                                                                          \
            " -s " + model.path());                                                                                                 \
          ASSERT_EQ(train.exit_code, 0) << "train failed";                                                                          \
                                                                                                                                    \
          auto model_json = json::parse(model.read());                                                                              \
          ASSERT_TRUE(model_json.contains("oob_error"));                                                                            \
          EXPECT_NEAR(model_json["oob_error"].get<double>(), golden["oob_error"].get<double>(), 1e-3) << #TestName " oob_error";    \
                                                                                                                                    \
          ASSERT_TRUE(model_json.contains("variable_importance"));                                                                  \
          auto vi  = model_json["variable_importance"];                                                                             \
          auto gvi = golden["variable_importance"];                                                                                 \
          compare_float_array(vi["scale"], gvi["scale"], 1e-3, #TestName " VI scale");                                              \
          compare_float_array(vi["projections"], gvi["projections"], 1e-3, #TestName " VI projections");                            \
          compare_float_array(vi["weighted_projections"], gvi["weighted_projections"], 1e-3, #TestName " VI weighted_projections"); \
          compare_float_array(vi["permuted"], gvi["permuted"], 1e-3, #TestName " VI permuted");                                     \
                                                                                                                                    \
          TempFile output;                                                                                                          \
          output.clear();                                                                                                           \
          auto predict = run_pptree("-q predict -M " + model.path() + " -d " + csv + " -o " + output.path());                       \
          ASSERT_EQ(predict.exit_code, 0) << "predict failed";                                                                      \
                                                                                                                                    \
          auto pred_json = json::parse(output.read());                                                                              \
          compare_predictions(pred_json["predictions"], golden["predictions"]);                                                     \
          EXPECT_NEAR(pred_json["error_rate"].get<double>(), golden["error_rate"].get<double>(), 1e-3) << #TestName " error_rate";  \
          compare_confusion_matrix(pred_json["confusion_matrix"], golden["confusion_matrix"]);                                      \
        }

// ---------------------------------------------------------------------------
// Train subcommand
// ---------------------------------------------------------------------------

/* Basic forest training on iris data succeeds. */
TEST(CLIIntegration, TrainWithIrisData) {
  TempDir dir;
  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 -s " + dir.file("model.json"));
  EXPECT_EQ(result.exit_code, 0);
}

/* Saved forest JSON contains model_type, model, and config block. */
TEST(CLIIntegration, TrainAndSaveForest) {
  TempFile model;
  model.clear();
  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["model_type"], "forest");
  EXPECT_TRUE(j.contains("model"));
  EXPECT_TRUE(j.contains("config"));

  auto config = j["config"];
  EXPECT_EQ(config["trees"], 5);
  EXPECT_TRUE(config.contains("lambda"));
  EXPECT_EQ(config["seed"], 42);
  EXPECT_TRUE(config.contains("threads"));
  EXPECT_TRUE(config.contains("vars"));
  EXPECT_EQ(config["data"], IRIS_CSV);
}

/* Single tree (t=0) saves with model_type "tree". */
TEST(CLIIntegration, TrainAndSaveSingleTree) {
  TempFile model;
  model.clear();
  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["model_type"], "tree");
}

/* Training writes a model file to the specified path. */
TEST(CLIIntegration, TrainDefaultSavesModel) {
  TempDir dir;
  // Run train from inside temp dir so default model.json goes there
  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 -s " + dir.file("model.json"));
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_TRUE(std::filesystem::exists(dir.file("model.json")));
}

/* --no-save suppresses model file creation. */
TEST(CLIIntegration, TrainNoSave) {
  TempDir dir;
  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 --no-save");
  EXPECT_EQ(result.exit_code, 0);
}

/* Saving to an existing file must fail (no silent overwrite). */
TEST(CLIIntegration, TrainCollisionFails) {
  TempFile model;
  // Don't clear - file exists, should fail
  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 -s " + model.path());
  EXPECT_NE(result.exit_code, 0);
}

/* Train without a data file must fail. */
TEST(CLIIntegration, TrainMissingDataFails) {
  auto result = run_pptree("train");
  EXPECT_NE(result.exit_code, 0);
}

/* Train with a nonexistent data file must fail. */
TEST(CLIIntegration, TrainNonexistentFileFails) {
  auto result = run_pptree("train -d /nonexistent/file.csv");
  EXPECT_NE(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Variable importance (always-on for forests, disabled by --no-metrics)
// ---------------------------------------------------------------------------

/* Forest training prints OOB error and Variable Importance table by default. */
TEST(CLIIntegration, TrainVIShownByDefault) {
  auto result = run_pptree("train -d " + IRIS_CSV + " -t 5 -r 42 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("OOB error"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Variable Importance"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Projection"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Weighted"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Permuted"), std::string::npos);
}

/* --no-metrics suppresses OOB error and Variable Importance table. */
TEST(CLIIntegration, TrainNoMetricsSuppressesVI) {
  auto result = run_pptree("train -d " + IRIS_CSV + " -t 5 -r 42 --no-save --no-metrics");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("OOB error"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Variable Importance"), std::string::npos);
}

/* Saved forest JSON contains oob_error and variable_importance. */
TEST(CLIIntegration, TrainVISavedToJson) {
  TempFile model;
  model.clear();
  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());

  ASSERT_TRUE(j.contains("oob_error")) << "Expected oob_error key in saved JSON";
  EXPECT_TRUE(j["oob_error"].is_number());
  EXPECT_GE(j["oob_error"].get<double>(), 0.0);
  EXPECT_LE(j["oob_error"].get<double>(), 1.0);

  ASSERT_TRUE(j.contains("variable_importance")) << "Expected variable_importance key in saved JSON";

  auto vi = j["variable_importance"];
  EXPECT_TRUE(vi.contains("scale"));
  EXPECT_TRUE(vi.contains("projections"));
  EXPECT_TRUE(vi.contains("weighted_projections"));
  EXPECT_TRUE(vi.contains("permuted"));

  EXPECT_EQ(vi["scale"].size(), 4u);
  EXPECT_EQ(vi["projections"].size(), 4u);
  EXPECT_EQ(vi["weighted_projections"].size(), 4u);
  EXPECT_EQ(vi["permuted"].size(), 4u);
}

/* With --no-metrics the saved JSON must not contain oob_error or variable_importance. */
TEST(CLIIntegration, TrainNoMetricsNotInJson) {
  TempFile model;
  model.clear();
  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 --no-metrics -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_FALSE(j.contains("oob_error"));
  EXPECT_FALSE(j.contains("variable_importance"));
}

/* Single tree training shows VI2 (projections) only, no OOB error. */
TEST(CLIIntegration, TrainSingleTreeShowsVI2Only) {
  auto result = run_pptree("train -d " + IRIS_CSV + " -t 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("OOB error"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Variable Importance"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Projection"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Weighted"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Permuted"), std::string::npos);
}

/* Single tree saved JSON contains only scale and projections (no weighted/permuted). */
TEST(CLIIntegration, TrainSingleTreeVISavedToJson) {
  TempFile model;
  model.clear();
  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  ASSERT_TRUE(j.contains("variable_importance"));

  auto vi = j["variable_importance"];
  EXPECT_TRUE(vi.contains("scale"));
  EXPECT_TRUE(vi.contains("projections"));
  EXPECT_FALSE(vi.contains("weighted_projections"));
  EXPECT_FALSE(vi.contains("permuted"));

  EXPECT_EQ(vi["scale"].size(), 4u);
  EXPECT_EQ(vi["projections"].size(), 4u);
}

/* Quiet mode suppresses the VI table but still saves it to JSON. */
TEST(CLIIntegration, TrainVIQuietSuppressesTable) {
  TempFile model;
  model.clear();
  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("Variable Importance"), std::string::npos)
    << "Expected no VI table in quiet mode";

  // JSON must still contain VI.
  auto j = json::parse(model.read());
  EXPECT_TRUE(j.contains("variable_importance"));
}

// ---------------------------------------------------------------------------
// Predict subcommand (fixture-based)
// ---------------------------------------------------------------------------

/**
 * @brief Test fixture that trains a forest model in SetUp().
 *
 * Every test using TEST_F(PredictTest, ...) starts with a trained
 * 5-tree forest saved to a temporary model file, ready for predict
 * invocations.
 */
class PredictTest : public ::testing::Test {
  protected:
    void SetUp() override {
      model_.reset(new TempFile());
      model_->clear();
      auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 -s " + model_->path());
      ASSERT_EQ(result.exit_code, 0);
    }

    std::unique_ptr<TempFile> model_;
};

/* Default predict shows error rate and confusion matrix. */
TEST_F(PredictTest, PredictWithSavedModel) {
  auto result = run_pptree("predict -M " + model_->path() + " -d " + IRIS_CSV);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_FALSE(result.stdout_output.empty());
  EXPECT_NE(result.stdout_output.find("Error rate:"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Confusion Matrix:"), std::string::npos);
}

/* --no-metrics in quiet mode suppresses error rate and confusion matrix. */
TEST_F(PredictTest, PredictNoMetrics) {
  auto result = run_pptree("-q predict -M " + model_->path() + " -d " + IRIS_CSV + " --no-metrics");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("Error rate:"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Confusion Matrix:"), std::string::npos);
}

/* --no-metrics without quiet still suppresses metrics output. */
TEST_F(PredictTest, PredictNoMetricsWithoutQuiet) {
  // --no-metrics without -q should also suppress metrics
  auto result = run_pptree("--no-color predict -M " + model_->path() + " -d " + IRIS_CSV + " --no-metrics");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("Error rate:"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Confusion Matrix:"), std::string::npos);
}

/* --no-metrics omits error_rate and confusion_matrix from JSON output. */
TEST_F(PredictTest, PredictNoMetricsOutputFile) {
  // --no-metrics should omit error_rate and confusion_matrix from output file
  TempFile output;
  output.clear();
  auto result = run_pptree("-q predict -M " + model_->path() + " -d " + IRIS_CSV + " --no-metrics -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_FALSE(j.contains("error_rate"));
  EXPECT_FALSE(j.contains("confusion_matrix"));
}

/* Without --output, predict shows a hint about --output. */
TEST_F(PredictTest, PredictSuggestsOutputHint) {
  auto result = run_pptree("--no-color predict -M " + model_->path() + " -d " + IRIS_CSV);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("--output"), std::string::npos);
}

/* When --output is used, the hint is suppressed. */
TEST_F(PredictTest, PredictNoHintWhenOutputUsed) {
  TempFile output;
  output.clear();
  auto result = run_pptree("--no-color predict -M " + model_->path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("Tip:"), std::string::npos);
}

/* Quiet mode produces completely empty stdout. */
TEST_F(PredictTest, PredictQuietSuppressesAll) {
  auto result = run_pptree("-q predict -M " + model_->path() + " -d " + IRIS_CSV);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_TRUE(result.stdout_output.empty());
}

/* -o writes predictions, error_rate, and confusion_matrix to JSON. */
TEST_F(PredictTest, PredictOutputFile) {
  TempFile output;
  output.clear();
  auto result = run_pptree("-q predict -M " + model_->path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_TRUE(j.contains("error_rate"));
  EXPECT_TRUE(j.contains("confusion_matrix"));
}

/* Writing to an existing output file must fail. */
TEST_F(PredictTest, PredictOutputCollisionFails) {
  TempFile output;
  // Don't clear - file exists, should fail
  auto result = run_pptree("-q predict -M " + model_->path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_NE(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Predict subcommand — error cases
// ---------------------------------------------------------------------------

/* Predict without -M must fail. */
TEST(CLIIntegration, PredictMissingModelArgFails) {
  auto result = run_pptree("predict -d " + IRIS_CSV);
  EXPECT_NE(result.exit_code, 0);
}

/* Predict with a nonexistent model file must fail. */
TEST(CLIIntegration, PredictNonexistentModelFails) {
  auto result = run_pptree("predict -M /nonexistent.json -d " + IRIS_CSV);
  EXPECT_NE(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Evaluate subcommand
// ---------------------------------------------------------------------------

/* Basic evaluation with simulated data succeeds. */
TEST(CLIIntegration, EvaluateWithSimulatedData) {
  auto result = run_pptree("-q evaluate --simulate 50x3x2 -t 5 -r 42 -i 1");
  EXPECT_EQ(result.exit_code, 0);
}

/* Evaluation with real iris data succeeds. */
TEST(CLIIntegration, EvaluateWithIrisData) {
  auto result = run_pptree("-q evaluate -d " + IRIS_CSV + " -t 5 -r 42 -i 1");
  EXPECT_EQ(result.exit_code, 0);
}

/* Non-quiet evaluate prints results header and error metrics. */
TEST(CLIIntegration, EvaluateTextOutput) {
  auto result = run_pptree("evaluate --simulate 50x3x2 -t 5 -r 42 -i 1");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("Evaluation results"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Train Err"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Test Err"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Peak RSS"), std::string::npos);
}

/* Evaluation with a single tree (t=0) succeeds. */
TEST(CLIIntegration, EvaluateSingleTree) {
  auto result = run_pptree("-q evaluate --simulate 50x3x2 -t 0 -r 42 -i 1");
  EXPECT_EQ(result.exit_code, 0);
}

/* -o writes evaluation stats (runs, mean_time, peak_rss) to JSON. */
TEST(CLIIntegration, EvaluateOutputFile) {
  TempFile output;
  output.clear();
  auto result = run_pptree("-q evaluate --simulate 50x3x2 -t 5 -r 42 -i 1 -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("runs"));
  EXPECT_TRUE(j.contains("mean_time_ms"));
  EXPECT_TRUE(j.contains("peak_rss_bytes"));
}

/* -e exports config.json, data.csv, and results.json to a directory. */
TEST(CLIIntegration, EvaluateExport) {
  TempDir dir;
  std::string export_path = dir.path() + "/exp1";

  auto result = run_pptree("-q evaluate --simulate 50x3x2 -t 5 -r 42 -l 0.3 -i 1 -e " + export_path);
  EXPECT_EQ(result.exit_code, 0);

  EXPECT_TRUE(std::filesystem::exists(export_path + "/config.json"));
  EXPECT_TRUE(std::filesystem::exists(export_path + "/data.csv"));
  EXPECT_TRUE(std::filesystem::exists(export_path + "/results.json"));

  // Verify config.json captures all configuration for reproducibility
  std::ifstream config_in(export_path + "/config.json");
  auto config = json::parse(config_in);
  EXPECT_EQ(config["data"], "data.csv");
  EXPECT_EQ(config["trees"], 5);
  EXPECT_FLOAT_EQ(config["lambda"].get<float>(), 0.3f);
  EXPECT_EQ(config["seed"], 42);
  EXPECT_TRUE(config.contains("threads"));
  EXPECT_TRUE(config.contains("train-ratio"));
  EXPECT_TRUE(config.contains("iterations"));
}

/* Output to an existing file must fail. */
TEST(CLIIntegration, EvaluateOutputCollisionFails) {
  TempFile output;
  // Don't clear - file exists, should fail
  auto result = run_pptree("-q evaluate --simulate 50x3x2 -t 5 -r 42 -i 1 -o " + output.path());
  EXPECT_NE(result.exit_code, 0);
}

/* Export to an existing directory must fail. */
TEST(CLIIntegration, EvaluateExportCollisionFails) {
  TempDir dir;
  // dir.path() already exists, should fail
  auto result = run_pptree("-q evaluate --simulate 50x3x2 -t 5 -r 42 -i 1 -e " + dir.path());
  EXPECT_NE(result.exit_code, 0);
}

/* JSON output includes an iterations array with per-run metrics. */
TEST(CLIIntegration, EvaluateOutputHasIterationsArray) {
  TempFile output;
  output.clear();
  auto result = run_pptree("-q evaluate --simulate 50x3x2 -t 5 -r 42 -i 1 -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("iterations"));
  EXPECT_EQ(j["iterations"].size(), 1u);
  EXPECT_TRUE(j["iterations"][0].contains("train_time_ms"));
  EXPECT_TRUE(j["iterations"][0].contains("train_error"));
  EXPECT_TRUE(j["iterations"][0].contains("test_error"));
}

/* Multiple iterations (-i 3) produce matching array size. */
TEST(CLIIntegration, EvaluateMultipleIterationsArray) {
  TempFile output;
  output.clear();
  auto result = run_pptree("-q evaluate --simulate 50x3x2 -t 5 -r 42 -i 3 -o " + output.path());
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
TEST(CLIIntegration, EvaluateNoDataSourceFails) {
  auto result = run_pptree("evaluate");
  EXPECT_NE(result.exit_code, 0);
}

/* Malformed --simulate string must fail. */
TEST(CLIIntegration, EvaluateInvalidSimFormatFails) {
  auto result = run_pptree("evaluate --simulate 100x5");
  EXPECT_NE(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Global flags and help
// ---------------------------------------------------------------------------

/* --help prints subcommand names and exits successfully. */
TEST(CLIIntegration, HelpFlag) {
  auto result = run_pptree("--help");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("train"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("predict"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("evaluate"), std::string::npos);
}

/* --version prints non-empty version string and exits successfully. */
TEST(CLIIntegration, VersionFlag) {
  auto result = run_pptree("--version");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_FALSE(result.stdout_output.empty());
}

/* -q with evaluate produces completely empty stdout. */
TEST(CLIIntegration, QuietSuppressesOutput) {
  auto quiet_result = run_pptree("-q evaluate --simulate 50x3x2 -t 5 -r 42 -i 1");
  EXPECT_EQ(quiet_result.exit_code, 0);
  EXPECT_TRUE(quiet_result.stdout_output.empty());
}

/* -q suppresses "Evaluation results", "Train Error", "Test Error". */
TEST(CLIIntegration, QuietSuppressesEvaluateResults) {
  auto quiet_result = run_pptree("-q evaluate --simulate 50x3x2 -t 5 -r 42 -i 1");
  EXPECT_EQ(quiet_result.exit_code, 0);
  EXPECT_EQ(quiet_result.stdout_output.find("Evaluation results"), std::string::npos);
  EXPECT_EQ(quiet_result.stdout_output.find("Train Error"), std::string::npos);
  EXPECT_EQ(quiet_result.stdout_output.find("Test Error"), std::string::npos);
}

/* No arguments at all exits with a non-zero code. */
TEST(CLIIntegration, NoArgsExitsNonZero) {
  auto result = run_pptree("");
  EXPECT_NE(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Config File
// ---------------------------------------------------------------------------

/* A JSON config file overrides default parameters. */
TEST(CLIIntegration, ConfigFileApplied) {
  TempFile config;
  {
    std::ofstream out(config.path());
    out << R"({"trees": 3})";
  }

  TempFile output;
  output.clear();
  auto result = run_pptree("--config " + config.path() + " -q evaluate --simulate 50x3x2 -r 42 -i 1 -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("runs"));
}

// ---------------------------------------------------------------------------
// End-to-End Pipeline
// ---------------------------------------------------------------------------

/* Full pipeline: train a forest, then predict on the same data. */
TEST(CLIIntegration, TrainThenPredict) {
  TempFile model;
  model.clear();

  auto train_result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 -s " + model.path());
  ASSERT_EQ(train_result.exit_code, 0);

  TempFile output;
  output.clear();
  auto predict_result = run_pptree("-q predict -M " + model.path() + " -d " + IRIS_CSV + " -o " + output.path());
  ASSERT_EQ(predict_result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_GT(j["predictions"].size(), 0u);
  EXPECT_TRUE(j.contains("error_rate"));
  EXPECT_TRUE(j.contains("confusion_matrix"));

  for (const auto& pred : j["predictions"]) {
    EXPECT_TRUE(pred.is_number_integer());
  }
}

// ---------------------------------------------------------------------------
// Vars parsing — fraction syntax
// ---------------------------------------------------------------------------

/* Fraction "1/3" is accepted end-to-end for evaluate. */
TEST(CLIIntegration, EvaluateWithFractionVars) {
  auto result = run_pptree("-q evaluate --simulate 50x3x2 -t 5 -r 42 -i 1 -v 1/3");
  EXPECT_EQ(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Single tree — config structure
// ---------------------------------------------------------------------------

/* Single tree config omits the "vars" key (not applicable). */
TEST(CLIIntegration, TrainSingleTreeConfigNoVars) {
  TempFile model;
  model.clear();
  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["model_type"], "tree");
  EXPECT_EQ(j["config"]["trees"], 0);
  // Single tree should not have vars in config
  EXPECT_FALSE(j["config"].contains("vars"));
}

// ---------------------------------------------------------------------------
// Confusion matrix JSON — no top-level "error" key
// ---------------------------------------------------------------------------

/* confusion_matrix in predict output must not contain "error". */
TEST(CLIIntegration, PredictOutputNoErrorInConfusionMatrix) {
  TempFile model;
  model.clear();
  auto train = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto predict = run_pptree("-q predict -M " + model.path() + " -d " + IRIS_CSV + " -o " + output.path());
  ASSERT_EQ(predict.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("confusion_matrix"));
  // confusion_matrix should not have an "error" key (error_rate is at top level)
  EXPECT_FALSE(j["confusion_matrix"].contains("error"));
}

// ---------------------------------------------------------------------------
// Automatic .json extension
// ---------------------------------------------------------------------------

/* Saving without .json extension auto-appends it. */
TEST(CLIIntegration, TrainAutoAppendsJsonExtension) {
  TempDir dir;
  std::string path_no_ext = dir.file("mymodel");

  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 -s " + path_no_ext);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_TRUE(std::filesystem::exists(path_no_ext + ".json"));
}

// ---------------------------------------------------------------------------
// CLI Golden Tests — end-to-end reproducibility through CLI pipeline
// ---------------------------------------------------------------------------

CLI_GOLDEN_TREE_TEST(IrisTreeGLDA,   "iris", "tree-glda-s42",   IRIS_CSV, 0.0f, 42)
CLI_GOLDEN_TREE_TEST(CrabTreeGLDA,   "crab", "tree-glda-s42",   CRAB_CSV, 0.0f, 42)

CLI_GOLDEN_FOREST_TEST(IrisForestGLDA,  "iris",  "forest-glda-t5-s42",      IRIS_CSV,  5,  0.0f, 2, 42)
CLI_GOLDEN_FOREST_TEST(IrisForestPDA,   "iris",  "forest-pda-l05-t5-s42",   IRIS_CSV,  5,  0.5f, 2, 42)
CLI_GOLDEN_FOREST_TEST(CrabForestGLDA,  "crab",  "forest-glda-t10-s42",     CRAB_CSV,  10, 0.0f, 3, 42)
CLI_GOLDEN_FOREST_TEST(WineForestGLDA,  "wine",  "forest-glda-t10-s42",     WINE_CSV,  10, 0.0f, 4, 42)
CLI_GOLDEN_FOREST_TEST(GlassForestGLDA, "glass", "forest-glda-t10-s42",     GLASS_CSV, 10, 0.0f, 3, 42)

// ---------------------------------------------------------------------------
// Parameter coverage — lambda, vars, config override
// ---------------------------------------------------------------------------

/* Training with explicit lambda saves it to config. */
TEST(CLIIntegration, TrainLambdaSaved) {
  TempFile model;
  model.clear();
  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 -l 0.5 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_FLOAT_EQ(j["config"]["lambda"].get<float>(), 0.5f);
}

/* Training with explicit vars saves it to config. */
TEST(CLIIntegration, TrainVarsSaved) {
  TempFile model;
  model.clear();
  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 -v 2 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["vars"], 2);
}

/* CLI args override config file values. */
TEST(CLIIntegration, CLIArgOverridesConfig) {
  TempFile config;
  {
    std::ofstream out(config.path());
    out << R"({"trees": 3, "seed": 99})";
  }

  TempFile model;
  model.clear();
  auto result = run_pptree("--config " + config.path() + " -q train -d " + IRIS_CSV + " -t 7 -r 42 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["trees"], 7);
  EXPECT_EQ(j["config"]["seed"], 42);
}

// ---------------------------------------------------------------------------
// Predict with single tree model
// ---------------------------------------------------------------------------

/* Predicting with a single-tree model produces correct output. */
TEST(CLIIntegration, PredictWithSingleTreeModel) {
  TempFile model;
  model.clear();
  auto train = run_pptree("-q train -d " + IRIS_CSV + " -t 0 -r 42 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto result = run_pptree("-q predict -M " + model.path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_EQ(j["predictions"].size(), 150u);
  EXPECT_TRUE(j.contains("error_rate"));
  EXPECT_TRUE(j.contains("confusion_matrix"));
}

// ---------------------------------------------------------------------------
// Datasets beyond iris — structural tests with crab and wine
// ---------------------------------------------------------------------------

/* Train and predict on crab data succeeds. */
TEST(CLIIntegration, TrainPredictCrab) {
  TempFile model;
  model.clear();
  auto train = run_pptree("-q train -d " + CRAB_CSV + " -t 5 -r 42 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto predict = run_pptree("-q predict -M " + model.path() + " -d " + CRAB_CSV + " -o " + output.path());
  EXPECT_EQ(predict.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_GT(j["predictions"].size(), 0u);
  EXPECT_TRUE(j.contains("error_rate"));
  EXPECT_TRUE(j.contains("confusion_matrix"));
}

/* Train and predict on wine data succeeds. */
TEST(CLIIntegration, TrainPredictWine) {
  TempFile model;
  model.clear();
  auto train = run_pptree("-q train -d " + WINE_CSV + " -t 5 -r 42 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto predict = run_pptree("-q predict -M " + model.path() + " -d " + WINE_CSV + " -o " + output.path());
  EXPECT_EQ(predict.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_GT(j["predictions"].size(), 0u);
  EXPECT_TRUE(j.contains("error_rate"));
  EXPECT_TRUE(j.contains("confusion_matrix"));
}

// ---------------------------------------------------------------------------
// Benchmark subcommand
// ---------------------------------------------------------------------------

static const std::string MINIMAL_SCENARIOS = R"({
  "name": "integration-test",
  "defaults": {
    "train_ratio": 0.7,
    "seed": 42,
    "lambda": 0.5,
    "iterations": 1
  },
  "scenarios": [
    { "name": "tiny-forest", "n": 50, "p": 3, "g": 2, "trees": 5, "vars": 0.5 },
    { "name": "tiny-tree",   "n": 50, "p": 3, "g": 2, "trees": 0 }
  ]
})";

static TempFile write_scenarios() {
  TempFile f;
  {
    std::ofstream out(f.path());
    out << MINIMAL_SCENARIOS;
  }
  return f;
}

/* Benchmark runs successfully with a scenarios file. */
TEST(CLIIntegration, BenchmarkRunsSuccessfully) {
  auto scenarios = write_scenarios();
  auto result    = run_pptree("-q --no-color benchmark -s " + scenarios.path());
  EXPECT_EQ(result.exit_code, 0);
}

/* Benchmark without -s must fail. */
TEST(CLIIntegration, BenchmarkNoScenariosFails) {
  auto result = run_pptree("-q --no-color benchmark");
  EXPECT_NE(result.exit_code, 0);
}

/* Benchmark with invalid scenarios file must fail. */
TEST(CLIIntegration, BenchmarkInvalidScenariosFails) {
  TempFile bad;
  {
    std::ofstream out(bad.path());
    out << "not valid json";
  }
  auto result = run_pptree("-q --no-color benchmark -s " + bad.path());
  EXPECT_NE(result.exit_code, 0);
}

/* Benchmark -o produces valid JSON results. */
TEST(CLIIntegration, BenchmarkJsonOutput) {
  auto scenarios = write_scenarios();
  TempFile output;
  output.clear();

  auto result = run_pptree("-q --no-color benchmark -s " + scenarios.path() + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("suite_name"));
  EXPECT_TRUE(j.contains("timestamp"));
  EXPECT_TRUE(j.contains("total_time_ms"));
  EXPECT_TRUE(j.contains("results"));
  EXPECT_EQ(j["results"].size(), 2u);

  for (const auto& r : j["results"]) {
    EXPECT_TRUE(r.contains("name"));
    EXPECT_TRUE(r.contains("n"));
    EXPECT_TRUE(r.contains("p"));
    EXPECT_TRUE(r.contains("g"));
    EXPECT_TRUE(r.contains("trees"));
    EXPECT_TRUE(r.contains("mean_time_ms"));
    EXPECT_TRUE(r.contains("mean_test_error"));
  }
}

/* Benchmark --csv produces valid CSV with header and data rows. */
TEST(CLIIntegration, BenchmarkCsvOutput) {
  auto scenarios = write_scenarios();
  TempFile csv_out(".csv");
  csv_out.clear();

  auto result = run_pptree("-q --no-color benchmark -s " + scenarios.path() + " --csv " + csv_out.path());
  EXPECT_EQ(result.exit_code, 0);

  std::ifstream in(csv_out.path());
  std::string header;
  std::getline(in, header);
  EXPECT_NE(header.find("scenario"), std::string::npos);
  EXPECT_NE(header.find("mean_time_ms"), std::string::npos);

  // Two data rows (one per scenario)
  std::string line;
  int data_rows = 0;

  while (std::getline(in, line))
    if (!line.empty()) data_rows++;

  EXPECT_EQ(data_rows, 2);
}

/* Benchmark -b compares against a baseline without error. */
TEST(CLIIntegration, BenchmarkBaselineComparison) {
  auto scenarios = write_scenarios();

  // First run: produce baseline
  TempFile baseline;
  baseline.clear();
  auto run1 = run_pptree("-q --no-color benchmark -s " + scenarios.path() + " -o " + baseline.path());
  ASSERT_EQ(run1.exit_code, 0);

  // Second run: compare against baseline
  auto run2 = run_pptree("-q --no-color benchmark -s " + scenarios.path() + " -b " + baseline.path());
  EXPECT_EQ(run2.exit_code, 0);
}

/* Benchmark -i overrides iteration count. */
TEST(CLIIntegration, BenchmarkIterationOverride) {
  auto scenarios = write_scenarios();
  TempFile output;
  output.clear();

  auto result = run_pptree("-q --no-color benchmark -s " + scenarios.path() + " -i 2 -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  for (const auto& r : j["results"]) {
    EXPECT_EQ(r["runs"].get<int>(), 2);
  }
}

/* Benchmark with missing scenarios file must fail. */
TEST(CLIIntegration, BenchmarkMissingFileFails) {
  auto result = run_pptree("-q --no-color benchmark -s /nonexistent/path.json");
  EXPECT_NE(result.exit_code, 0);
}
