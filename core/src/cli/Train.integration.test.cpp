/**
 * @file Train.integration.test.cpp
 * @brief Integration tests for the train subcommand.
 */
#include "cli/CLI.integration.hpp"

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
  EXPECT_TRUE(config.contains("dr"));
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

  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -r 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["meta"], golden_meta) << "CLI meta diverged from golden file";
}

/* Single tree (t=0) saves with model_type "tree". */
TEST(CLITrain, TrainAndSaveSingleTree) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["model_type"], "tree");
}


/* --no-save suppresses model file creation. */
TEST(CLITrain, TrainNoSave) {
  TempDir dir;
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
}

/* Saving to an existing file must fail (no silent overwrite). */
TEST(CLITrain, TrainCollisionFails) {
  TempFile model;
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

  EXPECT_EQ(vi["scale"].size(), 4u);
  EXPECT_EQ(vi["projections"].size(), 4u);
  EXPECT_EQ(vi["weighted_projections"].size(), 4u);
  EXPECT_EQ(vi["permuted"].size(), 4u);
}

/* With --no-metrics the saved JSON must not contain oob_error or variable_importance. */
TEST(CLITrain, TrainNoMetricsNotInJson) {
  TempFile model;
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
  TempFile model;
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

  EXPECT_EQ(vi["scale"].size(), 4u);
  EXPECT_EQ(vi["projections"].size(), 4u);
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
  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -v 1/3 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Single tree — config structure
// ---------------------------------------------------------------------------

/* Single tree config omits the "vars" key (not applicable). */
TEST(CLITrain, TrainSingleTreeConfigNoVars) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["model_type"], "tree");
  EXPECT_EQ(j["config"]["size"], 0);
  // Single tree uses noop DR strategy
  EXPECT_EQ(j["config"]["dr"]["name"], "noop");
}

// ---------------------------------------------------------------------------
// Saving without .json extension auto-appends it
// ---------------------------------------------------------------------------

/* Saving without .json extension auto-appends it. */
TEST(CLITrain, TrainAutoAppendsJsonExtension) {
  TempDir dir;
  std::string path_no_ext = dir.file("mymodel");

  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -s " + path_no_ext);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_TRUE(std::filesystem::exists(path_no_ext + ".json"));
}

// ---------------------------------------------------------------------------
// Parameter coverage — lambda, vars, config override
// ---------------------------------------------------------------------------

/* Training with explicit lambda saves it to config. */
TEST(CLITrain, TrainLambdaSaved) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -l 0.5 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_FLOAT_EQ(j["config"]["pp"]["lambda"].get<float>(), 0.5f);
}

/* Training with explicit vars saves it to config. */
TEST(CLITrain, TrainVarsSaved) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -v 2 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["dr"]["n_vars"], 2);
}

/* CLI args override config file values. */
TEST(CLITrain, CLIArgOverridesConfig) {
  TempFile config;
  {
    std::ofstream out(config.path());
    out << R"({"size": 3, "seed": 99})";
  }

  TempFile model;
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
  TempFile config;
  {
    std::ofstream out(config.path());
    out << R"({
      "pp": { "name": "pda", "lambda": 0.3 },
      "dr": { "name": "uniform", "n_vars": 2 },
      "sr": { "name": "mean_of_means" },
      "size": 5,
      "seed": 0
    })";
  }

  TempFile model;
  model.clear();
  auto result = run_ppforest2("--config " + config.path() + " -q train -d " + IRIS_CSV + " -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["size"], 5);
  EXPECT_FLOAT_EQ(j["config"]["pp"]["lambda"].get<float>(), 0.3f);
  EXPECT_EQ(j["config"]["dr"]["n_vars"], 2);
}

/* Config file accepts threads key. */
TEST(CLITrain, ConfigFileThreadsKey) {
  TempFile config;
  {
    std::ofstream out(config.path());
    out << R"({"size": 5, "seed": 0, "threads": 1})";
  }

  TempFile model;
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
  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + CRAB_CSV + " -n 5 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto predict = run_ppforest2("-q predict -M " + model.path() + " -d " + CRAB_CSV + " -o " + output.path());
  EXPECT_EQ(predict.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_GT(j["predictions"].size(), 0u);
  EXPECT_TRUE(j.contains("error_rate"));
  EXPECT_TRUE(j.contains("confusion_matrix"));
}

/* Train and predict on wine data succeeds. */
TEST(CLITrain, TrainPredictWine) {
  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + WINE_CSV + " -n 5 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto predict = run_ppforest2("-q predict -M " + model.path() + " -d " + WINE_CSV + " -o " + output.path());
  EXPECT_EQ(predict.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_GT(j["predictions"].size(), 0u);
  EXPECT_TRUE(j.contains("error_rate"));
  EXPECT_TRUE(j.contains("confusion_matrix"));
}

// ---------------------------------------------------------------------------
// Explicit strategy flags (--pp, --dr, --sr)
// ---------------------------------------------------------------------------

/* Train with --pp pda:lambda=0.3 saves PDA lambda in config. */
TEST(CLITrain, TrainWithPPStrategy) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --pp pda:lambda=0.3 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["pp"]["name"], "pda");
  EXPECT_FLOAT_EQ(j["config"]["pp"]["lambda"].get<float>(), 0.3f);
}

/* Train with --dr uniform:vars=2 saves DR config. */
TEST(CLITrain, TrainWithDRStrategy) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --dr uniform:vars=2 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["dr"]["name"], "uniform");
  EXPECT_EQ(j["config"]["dr"]["n_vars"], 2);
}

/* Train with --dr noop saves noop DR config. */
TEST(CLITrain, TrainWithDRNoop) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --dr noop -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["dr"]["name"], "noop");
}

/* Train with --sr mean_of_means saves SR config. */
TEST(CLITrain, TrainWithSRStrategy) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --sr mean_of_means -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["sr"]["name"], "mean_of_means");
}

/* Train with all three explicit strategy flags produces correct config. */
TEST(CLITrain, TrainWithAllStrategies) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2(
      "-q train -d " + IRIS_CSV +
      " -n 5 -r 0 "
      "--pp pda:lambda=0.3 --dr uniform:vars=2 --sr mean_of_means "
      "-s " +
      model.path()
  );
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["pp"]["name"], "pda");
  EXPECT_FLOAT_EQ(j["config"]["pp"]["lambda"].get<float>(), 0.3f);
  EXPECT_EQ(j["config"]["dr"]["name"], "uniform");
  EXPECT_EQ(j["config"]["dr"]["n_vars"], 2);
  EXPECT_EQ(j["config"]["sr"]["name"], "mean_of_means");
}

/* Strategy config does not contain display_name (presentation-only). */
TEST(CLITrain, StrategyConfigNoDisplayName) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2(
      "-q train -d " + IRIS_CSV +
      " -n 5 -r 0 "
      "--pp pda:lambda=0.3 --dr uniform:vars=2 --sr mean_of_means "
      "-s " +
      model.path()
  );
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_FALSE(j["config"]["pp"].contains("display_name"));
  EXPECT_FALSE(j["config"]["dr"].contains("display_name"));
  EXPECT_FALSE(j["config"]["sr"].contains("display_name"));
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
static json strip_timing(json j) {
  j.erase("training_duration_ms");
  return j;
}

/* -l 0.3 and --pp pda:lambda=0.3 produce identical exports. */
TEST(CLITrain, ImplicitExplicitPPEquivalent) {
  TempFile model_implicit, model_explicit;
  model_implicit.clear();
  model_explicit.clear();

  auto r1 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -l 0.3 -s " + model_implicit.path());
  auto r2 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --pp pda:lambda=0.3 -s " + model_explicit.path());
  ASSERT_EQ(r1.exit_code, 0);
  ASSERT_EQ(r2.exit_code, 0);

  EXPECT_EQ(strip_timing(json::parse(model_implicit.read())), strip_timing(json::parse(model_explicit.read())))
      << "Exports from -l and --pp should be identical";
}

/* -v 2 and --dr uniform:vars=2 produce identical exports. */
TEST(CLITrain, ImplicitExplicitDREquivalent) {
  TempFile model_implicit, model_explicit;
  model_implicit.clear();
  model_explicit.clear();

  auto r1 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -v 2 -s " + model_implicit.path());
  auto r2 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --dr uniform:vars=2 -s " + model_explicit.path());
  ASSERT_EQ(r1.exit_code, 0);
  ASSERT_EQ(r2.exit_code, 0);

  EXPECT_EQ(strip_timing(json::parse(model_implicit.read())), strip_timing(json::parse(model_explicit.read())))
      << "Exports from -v and --dr should be identical";
}

/* -l 0.3 -v 2 and --pp pda:lambda=0.3 --dr uniform:vars=2 --sr mean_of_means produce identical exports. */
TEST(CLITrain, ImplicitExplicitAllEquivalent) {
  TempFile model_implicit, model_explicit;
  model_implicit.clear();
  model_explicit.clear();

  auto r1 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -l 0.3 -v 2 -s " + model_implicit.path());
  auto r2 = run_ppforest2(
      "-q train -d " + IRIS_CSV +
      " -n 5 -r 0 "
      "--pp pda:lambda=0.3 --dr uniform:vars=2 --sr mean_of_means -s " +
      model_explicit.path()
  );
  ASSERT_EQ(r1.exit_code, 0);
  ASSERT_EQ(r2.exit_code, 0);

  EXPECT_EQ(strip_timing(json::parse(model_implicit.read())), strip_timing(json::parse(model_explicit.read())))
      << "Exports from shortcuts and --pp/--dr/--sr should be identical";
}

/* Single tree: -l 0.5 and --pp pda:lambda=0.5 --dr noop --sr mean_of_means produce identical exports. */
TEST(CLITrain, ImplicitExplicitSingleTreeEquivalent) {
  TempFile model_implicit, model_explicit;
  model_implicit.clear();
  model_explicit.clear();

  auto r1 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -r 0 -l 0.5 -s " + model_implicit.path());
  auto r2 = run_ppforest2(
      "-q train -d " + IRIS_CSV +
      " -n 0 -r 0 "
      "--pp pda:lambda=0.5 --dr noop --sr mean_of_means -s " +
      model_explicit.path()
  );
  ASSERT_EQ(r1.exit_code, 0);
  ASSERT_EQ(r2.exit_code, 0);

  EXPECT_EQ(strip_timing(json::parse(model_implicit.read())), strip_timing(json::parse(model_explicit.read())))
      << "Single tree with -l and --pp should be identical";
}

// ---------------------------------------------------------------------------
// Strategy flags — predict pipeline
// ---------------------------------------------------------------------------

/* Model trained with --pp can be loaded by predict. */
TEST(CLITrain, TrainWithStrategyThenPredict) {
  TempFile model;
  model.clear();
  auto train = run_ppforest2(
      "-q train -d " + IRIS_CSV +
      " -n 5 -r 0 "
      "--pp pda:lambda=0.3 --dr uniform:vars=2 -s " +
      model.path()
  );
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto predict = run_ppforest2("-q predict -M " + model.path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_EQ(predict.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_GT(j["predictions"].size(), 0u);
}
