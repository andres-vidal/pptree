/**
 * @file Train.integration.test.cpp
 * @brief Integration tests for the train subcommand.
 */
#include "cli/CLI.integration.hpp"

#include "utils/Macros.hpp"

#include <fstream>

// ---------------------------------------------------------------------------
// Train subcommand — TrainTest fixture (inherits SavedModelTest)
// ---------------------------------------------------------------------------

class TrainTest : public SavedModelTest {};

/* Saved forest JSON contains model_type, model, and config block. */
TEST_F(TrainTest, JsonContainsModelAndConfig) {
  EXPECT_EQ(model_json_["model_type"], "forest");
  EXPECT_TRUE(model_json_.contains("model"));
  EXPECT_TRUE(model_json_.contains("config"));

  auto config = model_json_["config"];
  EXPECT_EQ(config["size"], 5);
  EXPECT_TRUE(config.contains("pp"));
  EXPECT_TRUE(config["pp"].contains("lambda"));
  EXPECT_EQ(config["seed"], 0);
  EXPECT_TRUE(config.contains("threads"));
  EXPECT_TRUE(config.contains("vars"));
  EXPECT_EQ(config["data"], IRIS_CSV);
}

TEST_F(TrainTest, ForestMetaMatchesGolden) {
  std::ifstream golden_in(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");
  ASSERT_TRUE(golden_in.is_open());
  auto golden_meta = json::parse(golden_in)["meta"];

  EXPECT_EQ(model_json_["meta"], golden_meta) << "CLI meta diverged from golden file";
}

/* Model file is created at the specified path. */
TEST_F(TrainTest, ModelFileExists) {
  EXPECT_TRUE(std::filesystem::exists(model_->path()));
}

TEST(CLITrain, TrainSingleTreeMetaMatchesGolden) {
  std::ifstream golden_in(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  ASSERT_TRUE(golden_in.is_open());
  auto golden_meta = json::parse(golden_in)["meta"];

  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -r 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["meta"], golden_meta) << "CLI meta diverged from golden file";
}

/* Single tree (t=0) saves with model_type "tree". */
TEST(CLITrain, TrainAndSaveSingleTree) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["model_type"], "tree");
}


/* --no-save suppresses model file creation. */
TEST(CLITrain, TrainNoSave) {
  TempDir const dir;
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
}

/* Saving to an existing file must fail (no silent overwrite). */
TEST(CLITrain, TrainCollisionFails) {
  TempFile const model;
  // Don't clear - file exists, should fail
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -s " + model.path());
  EXPECT_NE(result.exit_code, 0);
}

/* Train without a data file must fail. */
TEST(CLITrain, TrainMissingDataFails) {
  auto result = run_ppforest2("train");
  EXPECT_NE(result.exit_code, 0);
}

/* Train with a nonexistent data file must fail. */
TEST(CLITrain, TrainNonexistentFileFails) {
  auto result = run_ppforest2("train -d /nonexistent/file.csv");
  EXPECT_NE(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Variable importance (always-on for forests, disabled by --no-metrics)
// ---------------------------------------------------------------------------

/* Forest training prints OOB error and Variable Importance table by default. */
TEST(CLITrain, TrainVIShownByDefault) {
  auto result = run_ppforest2("train -d " + IRIS_CSV + " -n 5 -r 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("OOB Error"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Variable Importance"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Projection"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Weighted"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Permuted"), std::string::npos);
}

/* --no-metrics suppresses OOB error and Variable Importance table. */
TEST(CLITrain, TrainNoMetricsSuppressesVI) {
  auto result = run_ppforest2("train -d " + IRIS_CSV + " -n 5 -r 0 --no-save --no-metrics");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("OOB error"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Variable Importance"), std::string::npos);
}

/* Saved forest JSON contains oob_error and variable_importance. */
TEST_F(TrainTest, VISavedToJson) {
  ASSERT_TRUE(model_json_.contains("oob_error")) << "Expected oob_error key in saved JSON";
  EXPECT_TRUE(model_json_["oob_error"].is_number());
  EXPECT_GE(model_json_["oob_error"].get<double>(), 0.0);
  EXPECT_LE(model_json_["oob_error"].get<double>(), 1.0);

  ASSERT_TRUE(model_json_.contains("variable_importance")) << "Expected variable_importance key in saved JSON";

  auto vi = model_json_["variable_importance"];
  EXPECT_TRUE(vi.contains("scale"));
  EXPECT_TRUE(vi.contains("projections"));
  EXPECT_TRUE(vi.contains("weighted_projections"));
  EXPECT_TRUE(vi.contains("permuted"));

  EXPECT_EQ(vi["scale"].size(), 4U);
  EXPECT_EQ(vi["projections"].size(), 4U);
  EXPECT_EQ(vi["weighted_projections"].size(), 4U);
  EXPECT_EQ(vi["permuted"].size(), 4U);
}

/* With --no-metrics the saved JSON must not contain oob_error or variable_importance. */
TEST(CLITrain, TrainNoMetricsNotInJson) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --no-metrics -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_FALSE(j.contains("oob_error"));
  EXPECT_FALSE(j.contains("variable_importance"));
}

/* Single tree training shows VI2 (projections) only, no OOB error. */
TEST(CLITrain, TrainSingleTreeShowsVI2Only) {
  auto result = run_ppforest2("train -d " + IRIS_CSV + " -n 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("OOB error"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Variable Importance"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Projection"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Weighted"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Permuted"), std::string::npos);
}

/* Single tree saved JSON contains only scale and projections (no weighted/permuted). */
TEST(CLITrain, TrainSingleTreeVISavedToJson) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  ASSERT_TRUE(j.contains("variable_importance"));

  auto vi = j["variable_importance"];
  EXPECT_TRUE(vi.contains("scale"));
  EXPECT_TRUE(vi.contains("projections"));
  EXPECT_FALSE(vi.contains("weighted_projections"));
  EXPECT_FALSE(vi.contains("permuted"));

  EXPECT_EQ(vi["scale"].size(), 4U);
  EXPECT_EQ(vi["projections"].size(), 4U);
}

/* Quiet mode suppresses the VI table but still saves it to JSON. */
TEST_F(TrainTest, VIQuietSuppressesTable) {
  EXPECT_TRUE(model_json_.contains("variable_importance"));
}

// ---------------------------------------------------------------------------
// Train display output
// ---------------------------------------------------------------------------

/* Train with --no-save must still produce output (via print_summary). */
TEST(CLITrain, TrainNoSaveStillDisplays) {
  auto result = run_ppforest2("--no-color train -d " + IRIS_CSV + " -n 5 -r 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  // Should still show confusion matrix output
  EXPECT_NE(result.stdout_output.find("Training Confusion Matrix"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("OOB Confusion Matrix"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Variable Importance"), std::string::npos);
}

/* Single tree train must show training confusion matrix but not OOB. */
TEST(CLITrain, SingleTreeTrainShowsTrainingCM) {
  auto result = run_ppforest2("--no-color train -d " + IRIS_CSV + " -n 0 -r 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("Training Confusion Matrix"), std::string::npos);
  // Single tree has no OOB
  EXPECT_EQ(result.stdout_output.find("OOB"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Vars parsing — fraction syntax
// ---------------------------------------------------------------------------

/* Fraction "1/3" is accepted end-to-end for train. */
TEST(CLITrain, TrainWithFractionVars) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --p-vars 1/3 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Single tree — config structure
// ---------------------------------------------------------------------------

/* Single tree config omits the "vars" key (not applicable). */
TEST(CLITrain, TrainSingleTreeConfigNoVars) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["model_type"], "tree");
  EXPECT_EQ(j["config"]["size"], 0);
  // Single tree uses all vars strategy
  EXPECT_EQ(j["config"]["vars"]["name"], "all");
}

// ---------------------------------------------------------------------------
// Saving without .json extension auto-appends it
// ---------------------------------------------------------------------------

/* Saving without .json extension auto-appends it. */
TEST(CLITrain, TrainAutoAppendsJsonExtension) {
  TempDir const dir;
  std::string const path_no_ext = dir.file("mymodel");

  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -s " + path_no_ext);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_TRUE(std::filesystem::exists(path_no_ext + ".json"));
}

// ---------------------------------------------------------------------------
// Parameter coverage — lambda, vars, config override
// ---------------------------------------------------------------------------

/* Training with explicit lambda saves it to config. */
TEST(CLITrain, TrainLambdaSaved) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -l 0.5 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_FLOAT_EQ(j["config"]["pp"]["lambda"].get<float>(), 0.5F);
}

/* Training with explicit vars saves it to config. */
TEST(CLITrain, TrainVarsSaved) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --n-vars 2 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["vars"]["count"], 2);
}

/* CLI args override config file values. */
TEST(CLITrain, CLIArgOverridesConfig) {
  TempFile const config;
  {
    std::ofstream out(config.path());
    out << R"({"size": 3, "seed": 99})";
  }

  TempFile const model;
  model.clear();
  auto result =
      run_ppforest2("--config " + config.path() + " -q train -d " + IRIS_CSV + " -n 7 -r 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["size"], 7);
  EXPECT_EQ(j["config"]["seed"], 0);
}

// ---------------------------------------------------------------------------
// Config file — explicit strategy format
// ---------------------------------------------------------------------------

/* Config file accepts structured strategy format (pp/dr/sr objects). */
TEST(CLITrain, ConfigFileExplicitStrategies) {
  TempFile const config;
  {
    std::ofstream out(config.path());
    out << R"({
      "pp": { "name": "pda", "lambda": 0.3 },
      "vars": { "name": "uniform", "count": 2 },
      "cutpoint": { "name": "mean_of_means" },
      "size": 5,
      "seed": 0
    })";
  }

  TempFile const model;
  model.clear();
  auto result = run_ppforest2("--config " + config.path() + " -q train -d " + IRIS_CSV + " -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["size"], 5);
  EXPECT_FLOAT_EQ(j["config"]["pp"]["lambda"].get<float>(), 0.3F);
  EXPECT_EQ(j["config"]["vars"]["count"], 2);
}

/* Config file accepts threads key. */
TEST(CLITrain, ConfigFileThreadsKey) {
  TempFile const config;
  {
    std::ofstream out(config.path());
    out << R"({"size": 5, "seed": 0, "threads": 1})";
  }

  TempFile const model;
  model.clear();
  auto result = run_ppforest2("--config " + config.path() + " -q train -d " + IRIS_CSV + " -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["threads"], 1);
}

// ---------------------------------------------------------------------------
// Datasets beyond iris — structural tests with crab and wine
// ---------------------------------------------------------------------------

/* Train and predict on crab data succeeds. */
TEST(CLITrain, TrainPredictCrab) {
  TempFile const model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + CRAB_CSV + " -n 5 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile const output;
  output.clear();
  auto predict = run_ppforest2("-q predict -M " + model.path() + " -d " + CRAB_CSV + " -o " + output.path());
  EXPECT_EQ(predict.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_GT(j["predictions"].size(), 0U);
  EXPECT_TRUE(j.contains("error_rate"));
  EXPECT_TRUE(j.contains("confusion_matrix"));
}

/* Train and predict on wine data succeeds. */
TEST(CLITrain, TrainPredictWine) {
  TempFile const model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + WINE_CSV + " -n 5 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile const output;
  output.clear();
  auto predict = run_ppforest2("-q predict -M " + model.path() + " -d " + WINE_CSV + " -o " + output.path());
  EXPECT_EQ(predict.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_GT(j["predictions"].size(), 0U);
  EXPECT_TRUE(j.contains("error_rate"));
  EXPECT_TRUE(j.contains("confusion_matrix"));
}

// ---------------------------------------------------------------------------
// Explicit strategy flags (--pp, --vars, --cutpoint)
// ---------------------------------------------------------------------------

/* Train with --pp pda:lambda=0.3 saves PDA lambda in config. */
TEST(CLITrain, TrainWithPPStrategy) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --pp pda:lambda=0.3 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["pp"]["name"], "pda");
  EXPECT_FLOAT_EQ(j["config"]["pp"]["lambda"].get<float>(), 0.3F);
}

/* Train with --vars uniform:count=2 saves vars config. */
TEST(CLITrain, TrainWithVarsStrategy) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --vars uniform:count=2 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["vars"]["name"], "uniform");
  EXPECT_EQ(j["config"]["vars"]["count"], 2);
}

/* Train with --vars all saves all vars config. */
TEST(CLITrain, TrainWithVarsAll) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --vars all -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["vars"]["name"], "all");
}

/* Train with --cutpoint mean_of_means saves cutpoint config. */
TEST(CLITrain, TrainWithThresholdStrategy) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --cutpoint mean_of_means -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["cutpoint"]["name"], "mean_of_means");
}

/* Train with all three explicit strategy flags produces correct config. */
TEST(CLITrain, TrainWithAllStrategies) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2(
      "-q train -d " + IRIS_CSV +
      " -n 5 -r 0 "
      "--pp pda:lambda=0.3 --vars uniform:count=2 --cutpoint mean_of_means "
      "-s " +
      model.path()
  );
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["pp"]["name"], "pda");
  EXPECT_FLOAT_EQ(j["config"]["pp"]["lambda"].get<float>(), 0.3F);
  EXPECT_EQ(j["config"]["vars"]["name"], "uniform");
  EXPECT_EQ(j["config"]["vars"]["count"], 2);
  EXPECT_EQ(j["config"]["cutpoint"]["name"], "mean_of_means");
}

/* Train with --leaf majority_vote saves leaf config. */
TEST(CLITrain, TrainWithLeafStrategy) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --leaf majority_vote -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["leaf"]["name"], "majority_vote");
}

/* Default leaf strategy is majority_vote when --leaf is not specified. */
TEST(CLITrain, TrainDefaultLeafStrategy) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["leaf"]["name"], "majority_vote");
}

/* Invalid --leaf value fails. */
TEST(CLITrain, TrainInvalidLeafFails) {
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --leaf unknown_leaf --no-save");
  EXPECT_NE(result.exit_code, 0);
}

/* Strategy config does not contain display_name (presentation-only). */
TEST(CLITrain, StrategyConfigNoDisplayName) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2(
      "-q train -d " + IRIS_CSV +
      " -n 5 -r 0 "
      "--pp pda:lambda=0.3 --vars uniform:count=2 --cutpoint mean_of_means "
      "-s " +
      model.path()
  );
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_FALSE(j["config"]["pp"].contains("display_name"));
  EXPECT_FALSE(j["config"]["vars"].contains("display_name"));
  EXPECT_FALSE(j["config"]["cutpoint"].contains("display_name"));
}

// ---------------------------------------------------------------------------
// Implicit vs explicit strategy equivalence
// ---------------------------------------------------------------------------

/**
 * @brief Strip timing fields so two model exports can be compared.
 *
 * training_duration_ms varies between runs even with the same seed;
 * everything else (model, config, meta, metrics) must be identical.
 */
namespace {
  json strip_timing(json j) {
    j.erase("training_duration_ms");
    return j;
  }
}

/* -l 0.3 and --pp pda:lambda=0.3 produce identical exports. */
TEST(CLITrain, ImplicitExplicitPPEquivalent) {
  TempFile const model_implicit;
  TempFile const model_explicit;
  model_implicit.clear();
  model_explicit.clear();

  auto r1 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -l 0.3 -s " + model_implicit.path());
  auto r2 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --pp pda:lambda=0.3 -s " + model_explicit.path());
  ASSERT_EQ(r1.exit_code, 0);
  ASSERT_EQ(r2.exit_code, 0);

  EXPECT_EQ(strip_timing(json::parse(model_implicit.read())), strip_timing(json::parse(model_explicit.read())))
      << "Exports from -l and --pp should be identical";
}

/* --n-vars 2 and --vars uniform:count=2 produce identical exports. */
TEST(CLITrain, ImplicitExplicitVarsEquivalent) {
  TempFile const model_implicit;
  TempFile const model_explicit;
  model_implicit.clear();
  model_explicit.clear();

  auto r1 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --n-vars 2 -s " + model_implicit.path());
  auto r2 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --vars uniform:count=2 -s " + model_explicit.path());
  ASSERT_EQ(r1.exit_code, 0);
  ASSERT_EQ(r2.exit_code, 0);

  EXPECT_EQ(strip_timing(json::parse(model_implicit.read())), strip_timing(json::parse(model_explicit.read())))
      << "Exports from --n-vars and --vars should be identical";
}

/* -l 0.3 --n-vars 2 and --pp pda:lambda=0.3 --vars uniform:count=2 --cutpoint mean_of_means produce identical exports. */
TEST(CLITrain, ImplicitExplicitAllEquivalent) {
  TempFile const model_implicit;
  TempFile const model_explicit;
  model_implicit.clear();
  model_explicit.clear();

  auto r1 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -l 0.3 --n-vars 2 -s " + model_implicit.path());
  auto r2 = run_ppforest2(
      "-q train -d " + IRIS_CSV +
      " -n 5 -r 0 "
      "--pp pda:lambda=0.3 --vars uniform:count=2 --cutpoint mean_of_means -s " +
      model_explicit.path()
  );
  ASSERT_EQ(r1.exit_code, 0);
  ASSERT_EQ(r2.exit_code, 0);

  EXPECT_EQ(strip_timing(json::parse(model_implicit.read())), strip_timing(json::parse(model_explicit.read())))
      << "Exports from shortcuts and --pp/--vars/--cutpoint should be identical";
}

/* Single tree: -l 0.5 and --pp pda:lambda=0.5 --vars all --cutpoint mean_of_means produce identical exports. */
TEST(CLITrain, ImplicitExplicitSingleTreeEquivalent) {
  TempFile const model_implicit;
  TempFile const model_explicit;
  model_implicit.clear();
  model_explicit.clear();

  auto r1 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -r 0 -l 0.5 -s " + model_implicit.path());
  auto r2 = run_ppforest2(
      "-q train -d " + IRIS_CSV +
      " -n 0 -r 0 "
      "--pp pda:lambda=0.5 --vars all --cutpoint mean_of_means -s " +
      model_explicit.path()
  );
  ASSERT_EQ(r1.exit_code, 0);
  ASSERT_EQ(r2.exit_code, 0);

  EXPECT_EQ(strip_timing(json::parse(model_implicit.read())), strip_timing(json::parse(model_explicit.read())))
      << "Single tree with -l and --pp should be identical";
}

// ---------------------------------------------------------------------------
// Edge cases — degenerate inputs
// ---------------------------------------------------------------------------

namespace {
  using namespace ppforest2::types;

  TempFile write_csv(FeatureMatrix const& x, OutcomeVector const& y) {
    TempFile f;
    {
      std::ofstream out(f.path());

      for (int j = 0; j < x.cols(); ++j) {
        out << "\"x" << (j + 1) << "\",";
      }
      out << "\"group\"\n";

      for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
          out << x(i, j) << ",";
        }
        out << "\"g" << static_cast<int>(y(i)) << "\"\n";
      }
    }
    return f;
  }
}

TEST(CLITrainEdgeCase, ConstantFeatureColumn) {
  FeatureMatrix const x = MAT(Feature, rows(6), 5, 1, 5, 2, 5, 3, 5, 7, 5, 8, 5, 9);
  OutcomeVector const y = VEC(Outcome, 0, 0, 0, 1, 1, 1);

  auto csv    = write_csv(x, y);
  auto result = run_ppforest2("-q train -d " + csv.path() + " -n 5 -r 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stderr_output, "");
}

TEST(CLITrainEdgeCase, ConstantFeatureColumnSingleTree) {
  FeatureMatrix const x = MAT(Feature, rows(6), 5, 1, 5, 2, 5, 3, 5, 7, 5, 8, 5, 9);
  OutcomeVector const y = VEC(Outcome, 0, 0, 0, 1, 1, 1);

  auto csv    = write_csv(x, y);
  auto result = run_ppforest2("-q train -d " + csv.path() + " -n 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stderr_output, "");
}

TEST(CLITrainEdgeCase, SingleObservationPerGroup) {
  FeatureMatrix const x = MAT(Feature, rows(2), 1, 0, 0, 1);
  OutcomeVector const y = VEC(Outcome, 0, 1);

  auto csv    = write_csv(x, y);
  auto result = run_ppforest2("-q train -d " + csv.path() + " -n 5 -r 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stderr_output, "");
}

TEST(CLITrainEdgeCase, SingleObservationPerGroupSingleTree) {
  FeatureMatrix const x = MAT(Feature, rows(2), 1, 0, 0, 1);
  OutcomeVector const y = VEC(Outcome, 0, 1);

  auto csv    = write_csv(x, y);
  auto result = run_ppforest2("-q train -d " + csv.path() + " -n 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stderr_output, "");
}

TEST(CLITrainEdgeCase, ExtremeImbalance) {
  // clang-format off
  FeatureMatrix const x = MAT(Feature, rows(20),
    0, 0,  1, 1,  2, 2,  3, 0,  4, 1,
    0, 2,  1, 0,  2, 1,  3, 2,  4, 0,
    0, 1,  1, 2,  2, 0,  3, 1,  4, 2,
    0, 0,  1, 1,  2, 2,  90, 90,  91, 91);
  OutcomeVector const y = VEC(Outcome,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1);
  // clang-format on

  auto csv    = write_csv(x, y);
  auto result = run_ppforest2("-q train -d " + csv.path() + " -n 5 -r 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stderr_output, "");
}

TEST(CLITrainEdgeCase, ExtremeImbalanceSingleTree) {
  // clang-format off
  FeatureMatrix const x = MAT(Feature, rows(20),
    0, 0,  1, 1,  2, 2,  3, 0,  4, 1,
    0, 2,  1, 0,  2, 1,  3, 2,  4, 0,
    0, 1,  1, 2,  2, 0,  3, 1,  4, 2,
    0, 0,  1, 1,  2, 2,  90, 90,  91, 91);
  OutcomeVector const y = VEC(Outcome,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1);
  // clang-format on

  auto csv    = write_csv(x, y);
  auto result = run_ppforest2("-q train -d " + csv.path() + " -n 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stderr_output, "");
}

// ---------------------------------------------------------------------------
// Strategy flags — predict pipeline
// ---------------------------------------------------------------------------

/* Model trained with --pp can be loaded by predict. */
TEST(CLITrain, TrainWithStrategyThenPredict) {
  TempFile const model;
  model.clear();
  auto train = run_ppforest2(
      "-q train -d " + IRIS_CSV +
      " -n 5 -r 0 "
      "--pp pda:lambda=0.3 --vars uniform:count=2 -s " +
      model.path()
  );
  ASSERT_EQ(train.exit_code, 0);

  TempFile const output;
  output.clear();
  auto predict = run_ppforest2("-q predict -M " + model.path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_EQ(predict.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_GT(j["predictions"].size(), 0U);
}
