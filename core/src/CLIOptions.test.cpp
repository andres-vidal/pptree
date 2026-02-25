/**
 * @file CLIOptions.test.cpp
 * @brief Unit tests for CLI argument parsing (parse_args), parameter
 *        initialization (init_params), unused-parameter warnings
 *        (warn_unused_params), ModelStats aggregation, and file helpers.
 *
 * Tests exercise the in-process parsing path by constructing argv arrays
 * and calling parse_args() directly.  Death tests (EXPECT_EXIT) verify
 * that invalid inputs cause a non-zero exit.
 */
#include <gtest/gtest.h>

#include "pptree.hpp"
#include "DataPacket.hpp"
#include "CLIOptions.hpp"
#include "Color.hpp"
#include "IO.hpp"

#include <filesystem>
#include <fstream>


class ThreadSafeDeathTests : public ::testing::Environment {
  void SetUp() override {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
  }
};

static auto * const kDeathTestEnv = ::testing::AddGlobalTestEnvironment(new ThreadSafeDeathTests);

using namespace pptree;


#ifndef PPTREE_DATA_DIR
#error "PPTREE_DATA_DIR must be defined"
#endif

static const std::string IRIS_PATH = std::string(PPTREE_DATA_DIR) + "/iris.csv";

/**
 * @brief Death-test predicate: matches any non-zero exit code.
 *
 * Google Test's built-in ExitedWithCode only matches a specific code.
 * This predicate accepts any non-zero exit, which is what we need
 * because CLI11 and std::exit may use different codes.
 */
class ExitedWithNonZero {
  public:
    bool operator()(int exit_status) const {
      #ifdef _WIN32
      return exit_status != 0;

      #else
      return testing::ExitedWithCode(0)(exit_status) == false && WIFEXITED(exit_status);

      #endif
    }
};

/**
 * @brief Build an argv vector and call parse_args().
 *
 * Convenience wrapper so tests can write:
 *   auto opts = parse({ "pptree", "train", "-d", "data.csv" });
 *
 * @param args_list Initializer list of C-string arguments (first must be program name).
 * @return Populated CLIOptions struct.
 */
static CLIOptions parse(std::initializer_list<const char *> args_list) {
  std::vector<const char *> args(args_list);
  return parse_args(static_cast<int>(args.size()), const_cast<char **>(args.data()));
}

// ---------------------------------------------------------------------------
// parse_args() — subcommand routing
// ---------------------------------------------------------------------------

/* Parsing "train" selects the train subcommand. */
TEST(ParseArgs, TrainSubcommand) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str() });
  EXPECT_EQ(opts.subcommand, Subcommand::train);
}

/* Parsing "evaluate" with --simulate selects the evaluate subcommand. */
TEST(ParseArgs, EvaluateSubcommandWithSimulate) {
  auto opts = parse({ "pptree", "evaluate", "--simulate", "100x5x2" });
  EXPECT_EQ(opts.subcommand, Subcommand::evaluate);
}

/* Parsing "evaluate" with -d selects evaluate and captures the data path. */
TEST(ParseArgs, EvaluateSubcommandWithData) {
  auto opts = parse({ "pptree", "evaluate", "-d", IRIS_PATH.c_str() });
  EXPECT_EQ(opts.subcommand, Subcommand::evaluate);
  EXPECT_EQ(opts.data_path, IRIS_PATH);
}

/* Omitting a subcommand must exit with a non-zero code. */
TEST(ParseArgs, NoSubcommandExits) {
  EXPECT_EXIT(
    parse({ "pptree" }),
    ExitedWithNonZero(),
    ""
    );
}

/* An unrecognised subcommand must exit with a non-zero code. */
TEST(ParseArgs, InvalidSubcommandExits) {
  EXPECT_EXIT(
    parse({ "pptree", "foobar" }),
    ExitedWithNonZero(),
    ""
    );
}

// ---------------------------------------------------------------------------
// parse_args() — train defaults and options
// ---------------------------------------------------------------------------

/* Verify all default values when only -d is supplied. */
TEST(ParseArgs, TrainDefaultValues) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str() });
  EXPECT_EQ(opts.trees, 100);
  EXPECT_FLOAT_EQ(opts.lambda, 0.5f);
  EXPECT_EQ(opts.threads, -1);
  EXPECT_EQ(opts.seed, -1);
  EXPECT_FLOAT_EQ(opts.p_vars, -1);
  EXPECT_EQ(opts.n_vars, -1);
  EXPECT_TRUE(opts.vars_input.empty());
  EXPECT_EQ(opts.save_path, "model.json");
  EXPECT_FALSE(opts.no_save);
}

/* -t overrides the default tree count. */
TEST(ParseArgs, TrainTreesOption) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-t", "50" });
  EXPECT_EQ(opts.trees, 50);
}

/* -l overrides the default lambda. */
TEST(ParseArgs, TrainLambdaOption) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-l", "0.3" });
  EXPECT_FLOAT_EQ(opts.lambda, 0.3f);
}

/* --threads sets the thread count explicitly. */
TEST(ParseArgs, TrainThreadsOption) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "--threads", "4" });
  EXPECT_EQ(opts.threads, 4);
}

/* -r sets the random seed. */
TEST(ParseArgs, TrainSeedOption) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-r", "42" });
  EXPECT_EQ(opts.seed, 42);
}

// ---------------------------------------------------------------------------
// parse_args() — vars (feature sub-sampling) parsing
// ---------------------------------------------------------------------------

/* A decimal value (0.8) is interpreted as a proportion. */
TEST(ParseArgs, VarsAsProportion) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-v", "0.8" });
  EXPECT_FLOAT_EQ(opts.p_vars, 0.8f);
  EXPECT_EQ(opts.n_vars, -1);
}

/* An integer (3) is interpreted as an absolute feature count. */
TEST(ParseArgs, VarsAsAbsoluteCount) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-v", "3" });
  EXPECT_EQ(opts.n_vars, 3);
  EXPECT_FLOAT_EQ(opts.p_vars, -1);
}

/* A fraction "1/3" is parsed as a proportion. */
TEST(ParseArgs, VarsAsFraction) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-v", "1/3" });
  EXPECT_NEAR(opts.p_vars, 1.0f / 3.0f, 0.001f);
  EXPECT_EQ(opts.n_vars, -1);
}

/* "1/2" parses to exactly 0.5. */
TEST(ParseArgs, VarsAsFractionHalf) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-v", "1/2" });
  EXPECT_FLOAT_EQ(opts.p_vars, 0.5f);
  EXPECT_EQ(opts.n_vars, -1);
}

/* "3/3" parses to 1.0 (use all features). */
TEST(ParseArgs, VarsAsFractionFull) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-v", "3/3" });
  EXPECT_FLOAT_EQ(opts.p_vars, 1.0f);
  EXPECT_EQ(opts.n_vars, -1);
}

/* Fraction > 1 (4/3) must exit. */
TEST(ParseArgs, VarsFractionGreaterThanOneExits) {
  EXPECT_EXIT(
    parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-v", "4/3" }),
    ExitedWithNonZero(),
    ""
    );
}

/* Division by zero (1/0) must exit. */
TEST(ParseArgs, VarsFractionZeroDenominatorExits) {
  EXPECT_EXIT(
    parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-v", "1/0" }),
    ExitedWithNonZero(),
    ""
    );
}

/* Zero numerator (0/3) must exit — no features selected. */
TEST(ParseArgs, VarsFractionZeroNumeratorExits) {
  EXPECT_EXIT(
    parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-v", "0/3" }),
    ExitedWithNonZero(),
    ""
    );
}

/* Non-numeric fraction ("a/b") must exit. */
TEST(ParseArgs, VarsFractionInvalidExits) {
  EXPECT_EXIT(
    parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-v", "a/b" }),
    ExitedWithNonZero(),
    ""
    );
}

// ---------------------------------------------------------------------------
// parse_args() — save, no-save, and lambda validation
// ---------------------------------------------------------------------------

/* -s sets a custom model output path. */
TEST(ParseArgs, TrainSaveOption) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-s", "/tmp/model.json" });
  EXPECT_EQ(opts.save_path, "/tmp/model.json");
}

/* Zero trees is accepted (means a single projection-pursuit tree). */
TEST(ParseArgs, TrainZeroTrees) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-t", "0" });
  EXPECT_EQ(opts.trees, 0);
}

/* Train without -d must exit. */
TEST(ParseArgs, TrainMissingDataExits) {
  EXPECT_EXIT(
    parse({ "pptree", "train" }),
    ExitedWithNonZero(),
    ""
    );
}

/* Train with a nonexistent data file must exit. */
TEST(ParseArgs, TrainNonexistentDataExits) {
  EXPECT_EXIT(
    parse({ "pptree", "train", "-d", "/nonexistent/file.csv" }),
    ExitedWithNonZero(),
    ""
    );
}

/* Lambda = 0 is a valid boundary value (pure LDA). */
TEST(ParseArgs, TrainLambdaZeroValid) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-l", "0" });
  EXPECT_FLOAT_EQ(opts.lambda, 0.0f);
}

/* Lambda = 1 is a valid boundary value (pure PCA). */
TEST(ParseArgs, TrainLambdaOneValid) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-l", "1.0" });
  EXPECT_FLOAT_EQ(opts.lambda, 1.0f);
}

/* Negative lambda must exit. */
TEST(ParseArgs, TrainLambdaNegativeExits) {
  EXPECT_EXIT(
    parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-l", "-0.1" }),
    ExitedWithNonZero(),
    ""
    );
}

/* Lambda > 1 must exit. */
TEST(ParseArgs, TrainLambdaOutOfRangeExits) {
  EXPECT_EXIT(
    parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-l", "2.0" }),
    ExitedWithNonZero(),
    ""
    );
}

/* --no-save disables model saving and clears the save path. */
TEST(ParseArgs, TrainNoSaveFlag) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "--no-save" });
  EXPECT_TRUE(opts.no_save);
  EXPECT_TRUE(opts.save_path.empty());
}

/* --no-save and -s together must exit (mutually exclusive). */
TEST(ParseArgs, TrainNoSaveExcludesSave) {
  EXPECT_EXIT(
    parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "--no-save", "-s", "/tmp/m.json" }),
    ExitedWithNonZero(),
    ""
    );
}

// ---------------------------------------------------------------------------
// parse_args() — predict options
// ---------------------------------------------------------------------------

/* -o sets the prediction output file path. */
TEST(ParseArgs, PredictOutputOption) {
  auto opts = parse({ "pptree", "predict", "-M", IRIS_PATH.c_str(), "-d", IRIS_PATH.c_str(), "-o", "/tmp/results.json" });
  EXPECT_EQ(opts.output_path, "/tmp/results.json");
}

/* --no-metrics flag is captured. */
TEST(ParseArgs, PredictNoMetricsFlag) {
  auto opts = parse({ "pptree", "predict", "-M", IRIS_PATH.c_str(), "-d", IRIS_PATH.c_str(), "--no-metrics" });
  EXPECT_TRUE(opts.no_metrics);
}

/* Predict without -M (model path) must exit. */
TEST(ParseArgs, PredictMissingModelExits) {
  EXPECT_EXIT(
    parse({ "pptree", "predict", "-d", IRIS_PATH.c_str() }),
    ExitedWithNonZero(),
    ""
    );
}

/* Predict without -d (data path) must exit. */
TEST(ParseArgs, PredictMissingDataExits) {
  EXPECT_EXIT(
    parse({ "pptree", "predict", "-M", IRIS_PATH.c_str() }),
    ExitedWithNonZero(),
    ""
    );
}

// ---------------------------------------------------------------------------
// parse_args() — evaluate options
// ---------------------------------------------------------------------------

/* -o sets the evaluation output file path. */
TEST(ParseArgs, EvaluateOutputOption) {
  auto opts = parse({ "pptree", "evaluate", "--simulate", "100x5x2", "-o", "/tmp/results.json" });
  EXPECT_EQ(opts.output_path, "/tmp/results.json");
}

/* Evaluate must reject -s (--save is train-only). */
TEST(ParseArgs, EvaluateNoSaveOption) {
  // Evaluate should not accept -s
  EXPECT_EXIT(
    parse({ "pptree", "evaluate", "--simulate", "100x5x2", "-s", "/tmp/model.json" }),
    ExitedWithNonZero(),
    ""
    );
}

/* Evaluate always has an empty save_path (save is not available). */
TEST(ParseArgs, EvaluateSavePathAlwaysEmpty) {
  auto opts = parse({ "pptree", "evaluate", "--simulate", "100x5x2" });
  EXPECT_TRUE(opts.save_path.empty());
}

/* -e sets the experiment export directory. */
TEST(ParseArgs, EvaluateExportOption) {
  auto opts = parse({ "pptree", "evaluate", "--simulate", "100x5x2", "-e", "/tmp/experiment" });
  EXPECT_EQ(opts.export_path, "/tmp/experiment");
}

/* --simulate "RxCxK" is parsed into rows, cols, and classes. */
TEST(ParseArgs, EvaluateSimulateFormat) {
  auto opts = parse({ "pptree", "evaluate", "--simulate", "100x5x2" });
  EXPECT_EQ(opts.rows, 100);
  EXPECT_EQ(opts.cols, 5);
  EXPECT_EQ(opts.classes, 2);
}

/* Simulation generation parameters (mean, separation, sd) are captured. */
TEST(ParseArgs, EvaluateSimulateCustomParams) {
  auto opts = parse({ "pptree", "evaluate", "--simulate", "100x5x3",
                      "--sim-mean", "200", "--sim-mean-separation", "25", "--sim-sd", "5" });
  EXPECT_FLOAT_EQ(opts.sim_mean, 200.0f);
  EXPECT_FLOAT_EQ(opts.sim_mean_separation, 25.0f);
  EXPECT_FLOAT_EQ(opts.sim_sd, 5.0f);
}

/* -p sets the train/test split ratio. */
TEST(ParseArgs, EvaluateTrainRatio) {
  auto opts = parse({ "pptree", "evaluate", "--simulate", "100x5x2", "-p", "0.8" });
  EXPECT_FLOAT_EQ(opts.train_ratio, 0.8f);
}

/* -i sets the number of evaluation iterations. */
TEST(ParseArgs, EvaluateIterations) {
  auto opts = parse({ "pptree", "evaluate", "--simulate", "100x5x2", "-i", "5" });
  EXPECT_EQ(opts.iterations, 5);
}

/* Malformed --simulate string (missing dimension) must exit. */
TEST(ParseArgs, EvaluateInvalidSimulateFormatExits) {
  EXPECT_EXIT(
    parse({ "pptree", "evaluate", "--simulate", "100x5" }),
    ExitedWithNonZero(),
    ""
    );
}

/* Simulation with only 1 class must exit (need >= 2 for classification). */
TEST(ParseArgs, EvaluateClassesMustBeGreaterThanOne) {
  EXPECT_EXIT(
    parse({ "pptree", "evaluate", "--simulate", "100x5x1" }),
    ExitedWithNonZero(),
    ""
    );
}

/* Evaluate without -d or --simulate must exit. */
TEST(ParseArgs, EvaluateNoDataSourceExits) {
  EXPECT_EXIT(
    parse({ "pptree", "evaluate" }),
    ExitedWithNonZero(),
    ""
    );
}

/* Providing both -d and --simulate must exit (mutually exclusive). */
TEST(ParseArgs, EvaluateBothDataAndSimulateExits) {
  EXPECT_EXIT(
    parse({ "pptree", "evaluate", "-d", IRIS_PATH.c_str(), "--simulate", "100x5x2" }),
    ExitedWithNonZero(),
    ""
    );
}

/* Simulation parameters without --simulate must exit. */
TEST(ParseArgs, EvaluateSimParamsNeedSimulate) {
  EXPECT_EXIT(
    parse({ "pptree", "evaluate", "-d", IRIS_PATH.c_str(), "--sim-mean", "200" }),
    ExitedWithNonZero(),
    ""
    );
}

// ---------------------------------------------------------------------------
// parse_args() — global flags
// ---------------------------------------------------------------------------

/* --no-color flag is captured when placed before the subcommand. */
TEST(ParseArgs, NoColorFlag) {
  auto opts = parse({ "pptree", "--no-color", "evaluate", "--simulate", "100x5x2" });
  EXPECT_TRUE(opts.no_color);
}

/* --no-color defaults to false. */
TEST(ParseArgs, NoColorDefaultFalse) {
  auto opts = parse({ "pptree", "evaluate", "--simulate", "100x5x2" });
  EXPECT_FALSE(opts.no_color);
}

/* -q enables quiet mode. */
TEST(ParseArgs, QuietShortFlag) {
  auto opts = parse({ "pptree", "-q", "evaluate", "--simulate", "100x5x2" });
  EXPECT_TRUE(opts.quiet);
}

/* --quiet also enables quiet mode. */
TEST(ParseArgs, QuietLongFlag) {
  auto opts = parse({ "pptree", "--quiet", "evaluate", "--simulate", "100x5x2" });
  EXPECT_TRUE(opts.quiet);
}


/* --version prints version info and exits with code 0. */
TEST(ParseArgs, VersionExits) {
  EXPECT_EXIT(
    parse({ "pptree", "--version" }),
    testing::ExitedWithCode(0),
    ""
    );
}

// ---------------------------------------------------------------------------
// init_params() — default resolution and validation
// ---------------------------------------------------------------------------

/* Sentinel lambda (-1) is replaced by the default 0.5. */
TEST(InitParams, DefaultLambda) {
  CLIOptions params;
  params.lambda = -1;
  params.quiet  = true;
  init_params(params);
  EXPECT_FLOAT_EQ(params.lambda, 0.5f);
}

/* An explicitly set lambda is preserved. */
TEST(InitParams, LambdaUnchangedIfSet) {
  CLIOptions params;
  params.lambda = 0.3f;
  params.quiet  = true;
  init_params(params);
  EXPECT_FLOAT_EQ(params.lambda, 0.3f);
}

/* Train ratio of 0 must exit (no training data). */
TEST(InitParams, InvalidTrainRatioZeroExits) {
  EXPECT_EXIT({
    CLIOptions params;
    params.train_ratio = 0;
    params.quiet       = true;
    init_params(params);
  }, ExitedWithNonZero(), "");
}

/* Train ratio of 1.0 must exit (no test data). */
TEST(InitParams, InvalidTrainRatioOneExits) {
  EXPECT_EXIT({
    CLIOptions params;
    params.train_ratio = 1.0f;
    params.quiet       = true;
    init_params(params);
  }, ExitedWithNonZero(), "");
}

/* Negative train ratio must exit. */
TEST(InitParams, InvalidTrainRatioNegativeExits) {
  EXPECT_EXIT({
    CLIOptions params;
    params.train_ratio = -0.5f;
    params.quiet       = true;
    init_params(params);
  }, ExitedWithNonZero(), "");
}

// ---------------------------------------------------------------------------
// init_params() — used_default_* tracking flags
// ---------------------------------------------------------------------------

/* Sentinel seed (-1) sets used_default_seed to true. */
TEST(InitParams, UsedDefaultSeedFlag) {
  CLIOptions params;
  params.seed  = -1;
  params.quiet = true;
  init_params(params);
  EXPECT_TRUE(params.used_default_seed);
}

/* An explicit seed clears used_default_seed. */
TEST(InitParams, UsedDefaultSeedFlagFalse) {
  CLIOptions params;
  params.seed  = 42;
  params.quiet = true;
  init_params(params);
  EXPECT_FALSE(params.used_default_seed);
}

/* Sentinel threads (-1) sets used_default_threads to true. */
TEST(InitParams, UsedDefaultThreadsFlag) {
  CLIOptions params;
  params.threads = -1;
  params.quiet   = true;
  init_params(params);
  EXPECT_TRUE(params.used_default_threads);
}

/* An explicit thread count clears used_default_threads. */
TEST(InitParams, UsedDefaultThreadsFlagFalse) {
  CLIOptions params;
  params.threads = 8;
  params.quiet   = true;
  init_params(params);
  EXPECT_FALSE(params.used_default_threads);
}

/* Sentinel vars (-1/-1) sets used_default_vars to true. */
TEST(InitParams, UsedDefaultVarsFlag) {
  CLIOptions params;
  params.trees  = 10;
  params.p_vars = -1;
  params.n_vars = -1;
  params.quiet  = true;
  init_params(params, 10);
  EXPECT_TRUE(params.used_default_vars);
}

/* An explicit p_vars clears used_default_vars. */
TEST(InitParams, UsedDefaultVarsFlagFalse) {
  CLIOptions params;
  params.trees  = 10;
  params.p_vars = 0.8f;
  params.n_vars = -1;
  params.quiet  = true;
  init_params(params, 10);
  EXPECT_FALSE(params.used_default_vars);
}

// ---------------------------------------------------------------------------
// init_params() — seed, threads, and vars auto-resolution
// ---------------------------------------------------------------------------

/* Sentinel seed (-1) is replaced by a generated seed. */
TEST(InitParams, AutoSeed) {
  CLIOptions params;
  params.seed  = -1;
  params.quiet = true;
  init_params(params);
  EXPECT_NE(params.seed, -1);
}

/* An explicit seed value is preserved. */
TEST(InitParams, SeedPreservedIfSet) {
  CLIOptions params;
  params.seed  = 42;
  params.quiet = true;
  init_params(params);
  EXPECT_EQ(params.seed, 42);
}

/* Sentinel threads (-1) auto-detects to >= 1 thread. */
TEST(InitParams, DefaultThreads) {
  CLIOptions params;
  params.threads = -1;
  params.quiet   = true;
  init_params(params);
  EXPECT_GE(params.threads, 1);
}

/* An explicitly set thread count is preserved. */
TEST(InitParams, ThreadsPreservedIfSet) {
  CLIOptions params;
  params.threads = 8;
  params.quiet   = true;
  init_params(params);
  EXPECT_EQ(params.threads, 8);
}

/* n_vars is computed from p_vars * total_vars. */
TEST(InitParams, NVarsFromPVars) {
  CLIOptions params;
  params.trees  = 10;
  params.p_vars = 0.5f;
  params.n_vars = -1;
  params.quiet  = true;
  init_params(params, 10);
  EXPECT_EQ(params.n_vars, 5);
}

/* p_vars is back-computed from n_vars / total_vars. */
TEST(InitParams, PVarsFromNVars) {
  CLIOptions params;
  params.trees  = 10;
  params.p_vars = -1;
  params.n_vars = 3;
  params.quiet  = true;
  init_params(params, 10);
  EXPECT_FLOAT_EQ(params.p_vars, 0.3f);
}

/* Default vars: p_vars = 0.5, n_vars = half of total. */
TEST(InitParams, DefaultPVarsAndNVars) {
  CLIOptions params;
  params.trees  = 10;
  params.p_vars = -1;
  params.n_vars = -1;
  params.quiet  = true;
  init_params(params, 10);
  EXPECT_FLOAT_EQ(params.p_vars, 0.5f);
  EXPECT_EQ(params.n_vars, 5);
}

/* Vars computation is skipped for a single tree (trees = 0). */
TEST(InitParams, NoVarsWhenSingleTree) {
  CLIOptions params;
  params.trees  = 0;
  params.p_vars = 0.8f;
  params.n_vars = -1;
  params.quiet  = true;
  init_params(params, 10);
  EXPECT_FLOAT_EQ(params.p_vars, 0.8f);
  EXPECT_EQ(params.n_vars, -1);
}

/* Vars computation is skipped when total_vars is 0 (unknown). */
TEST(InitParams, NoVarsWhenZeroTotalVars) {
  CLIOptions params;
  params.trees  = 10;
  params.p_vars = -1;
  params.n_vars = -1;
  params.quiet  = true;
  init_params(params, 0);
  EXPECT_FLOAT_EQ(params.p_vars, -1);
  EXPECT_EQ(params.n_vars, -1);
}

// ---------------------------------------------------------------------------
// warn_unused_params() — single-tree parameter warnings
// ---------------------------------------------------------------------------

/* Single tree with --threads warns that threads is ignored. */
TEST(WarnUnusedParams, TreesZeroThreadsWarning) {
  CLIOptions params;
  params.trees   = 0;
  params.threads = 4;
  params.quiet   = false;

  testing::internal::CaptureStdout();
  warn_unused_params(params);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("threads parameter is ignored"), std::string::npos);
}

/* Single tree with --vars warns that vars is ignored. */
TEST(WarnUnusedParams, TreesZeroVarsWarning) {
  CLIOptions params;
  params.trees  = 0;
  params.p_vars = 0.8f;
  params.quiet  = false;

  testing::internal::CaptureStdout();
  warn_unused_params(params);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("--vars parameter is ignored"), std::string::npos);
}

/* Single tree with both --threads and --vars emits both warnings. */
TEST(WarnUnusedParams, TreesZeroBothWarnings) {
  CLIOptions params;
  params.trees   = 0;
  params.threads = 4;
  params.n_vars  = 3;
  params.quiet   = false;

  testing::internal::CaptureStdout();
  warn_unused_params(params);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("threads parameter is ignored"), std::string::npos);
  EXPECT_NE(output.find("--vars parameter is ignored"), std::string::npos);
  EXPECT_NE(output.find("Single trees always use all features"), std::string::npos);
}

/* Forest mode (trees > 0) emits no warnings. */
TEST(WarnUnusedParams, TreesNonZeroNoWarning) {
  CLIOptions params;
  params.trees   = 10;
  params.threads = 4;
  params.p_vars  = 0.8f;
  params.quiet   = false;

  testing::internal::CaptureStdout();
  warn_unused_params(params);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_TRUE(output.empty());
}

/* Quiet mode suppresses all parameter warnings. */
TEST(WarnUnusedParams, QuietSuppresses) {
  CLIOptions params;
  params.trees   = 0;
  params.threads = 4;
  params.p_vars  = 0.8f;
  params.quiet   = true;

  testing::internal::CaptureStdout();
  warn_unused_params(params);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_TRUE(output.empty());
}


// ---------------------------------------------------------------------------
// ModelStats — aggregation and JSON serialization
// ---------------------------------------------------------------------------

/* Mean training time across runs. */
TEST(ModelStats, MeanTime) {
  ModelStats stats;
  stats.tr_times = Vector<float>(3);
  stats.tr_times << 10.0, 20.0, 30.0;

  EXPECT_FLOAT_EQ(stats.mean_time(), 20.0);
}

/* Mean training error across runs. */
TEST(ModelStats, MeanTrainError) {
  ModelStats stats;
  stats.tr_error = Vector<float>(3);
  stats.tr_error << 0.1, 0.2, 0.3;

  EXPECT_FLOAT_EQ(stats.mean_tr_error(), 0.2);
}

/* Mean test error across runs. */
TEST(ModelStats, MeanTestError) {
  ModelStats stats;
  stats.te_error = Vector<float>(3);
  stats.te_error << 0.05, 0.15, 0.25;

  EXPECT_FLOAT_EQ(stats.mean_te_error(), 0.15);
}

/* Standard deviation of training time is positive for varied inputs. */
TEST(ModelStats, StdTime) {
  ModelStats stats;
  stats.tr_times = Vector<float>(3);
  stats.tr_times << 10.0, 20.0, 30.0;

  EXPECT_GT(stats.std_time(), 0);
}

/* Standard deviation of training error is positive for varied inputs. */
TEST(ModelStats, StdTrainError) {
  ModelStats stats;
  stats.tr_error = Vector<float>(3);
  stats.tr_error << 0.1, 0.2, 0.3;

  EXPECT_GT(stats.std_tr_error(), 0);
}

/* Standard deviation of test error is positive for varied inputs. */
TEST(ModelStats, StdTestError) {
  ModelStats stats;
  stats.te_error = Vector<float>(3);
  stats.te_error << 0.05, 0.15, 0.25;

  EXPECT_GT(stats.std_te_error(), 0);
}

/* JSON output includes std_time_ms, std_train_error, std_test_error. */
TEST(ModelStats, StdFieldsInJson) {
  ModelStats stats;
  stats.tr_times = Vector<float>(2);
  stats.tr_times << 10.0, 20.0;
  stats.tr_error = Vector<float>(2);
  stats.tr_error << 0.1, 0.3;
  stats.te_error = Vector<float>(2);
  stats.te_error << 0.2, 0.4;

  auto j = stats.to_json();

  EXPECT_TRUE(j.contains("std_time_ms"));
  EXPECT_TRUE(j.contains("std_train_error"));
  EXPECT_TRUE(j.contains("std_test_error"));
  EXPECT_GT(j["std_time_ms"].get<double>(), 0);
  EXPECT_GT(j["std_train_error"].get<double>(), 0);
  EXPECT_GT(j["std_test_error"].get<double>(), 0);
}

/* A single run produces zero standard deviation for all metrics. */
TEST(ModelStats, SingleRunStdZero) {
  ModelStats stats;
  stats.tr_times = Vector<float>(1);
  stats.tr_times << 10.0;
  stats.tr_error = Vector<float>(1);
  stats.tr_error << 0.1;
  stats.te_error = Vector<float>(1);
  stats.te_error << 0.2;

  auto j = stats.to_json();

  EXPECT_EQ(j["runs"], 1);
  EXPECT_FLOAT_EQ(j["std_time_ms"].get<float>(), 0.0f);
  EXPECT_FLOAT_EQ(j["std_train_error"].get<float>(), 0.0f);
  EXPECT_FLOAT_EQ(j["std_test_error"].get<float>(), 0.0f);
}

/* Full JSON serialization: means, iterations array, no peak_rss when unset. */
TEST(ModelStats, ToJson) {
  ModelStats stats;
  stats.tr_times = Vector<float>(2);
  stats.tr_times << 10.0, 20.0;
  stats.tr_error = Vector<float>(2);
  stats.tr_error << 0.1, 0.3;
  stats.te_error = Vector<float>(2);
  stats.te_error << 0.2, 0.4;

  auto j = stats.to_json();

  EXPECT_EQ(j["runs"], 2);
  EXPECT_FLOAT_EQ(j["mean_time_ms"].get<float>(), 15.0);
  EXPECT_FLOAT_EQ(j["mean_train_error"].get<float>(), 0.2);
  EXPECT_FLOAT_EQ(j["mean_test_error"].get<float>(), 0.3);
  EXPECT_FALSE(j.contains("peak_rss_bytes"));

  // Per-iteration data
  EXPECT_TRUE(j.contains("iterations"));
  EXPECT_EQ(j["iterations"].size(), 2u);
  EXPECT_FLOAT_EQ(j["iterations"][0]["train_time_ms"].get<float>(), 10.0);
  EXPECT_FLOAT_EQ(j["iterations"][0]["train_error"].get<float>(), 0.1);
  EXPECT_FLOAT_EQ(j["iterations"][0]["test_error"].get<float>(), 0.2);
  EXPECT_FLOAT_EQ(j["iterations"][1]["train_time_ms"].get<float>(), 20.0);
  EXPECT_FLOAT_EQ(j["iterations"][1]["train_error"].get<float>(), 0.3);
  EXPECT_FLOAT_EQ(j["iterations"][1]["test_error"].get<float>(), 0.4);
}

/* JSON includes peak_rss_bytes/mb when set; iterations lack peak_rss. */
TEST(ModelStats, ToJsonWithRSS) {
  ModelStats stats;
  stats.tr_times = Vector<float>(1);
  stats.tr_times << 100.0;
  stats.tr_error = Vector<float>(1);
  stats.tr_error << 0.05;
  stats.te_error = Vector<float>(1);
  stats.te_error << 0.1;
  stats.peak_rss_bytes = 10485760L; // 10 MB

  auto j = stats.to_json();

  EXPECT_EQ(j["peak_rss_bytes"], 10485760L);
  EXPECT_NEAR(j["peak_rss_mb"].get<double>(), 10.0, 0.01);

  // Iterations should not contain peak_rss
  EXPECT_EQ(j["iterations"].size(), 1u);
  EXPECT_FALSE(j["iterations"][0].contains("peak_rss"));
}

// ---------------------------------------------------------------------------
// parse_args() — additional vars validation (edge cases)
// ---------------------------------------------------------------------------

/* Proportion of exactly 0.0 must exit — selects no features. */
TEST(ParseArgs, VarsProportionZeroExits) {
  EXPECT_EXIT(
    parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-v", "0.0" }),
    ExitedWithNonZero(),
    ""
    );
}

/* Proportion > 1.0 must exit. */
TEST(ParseArgs, VarsProportionOutOfRangeExits) {
  EXPECT_EXIT(
    parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-v", "1.5" }),
    ExitedWithNonZero(),
    ""
    );
}

/* Negative integer count must exit. */
TEST(ParseArgs, VarsNegativeCountExits) {
  EXPECT_EXIT(
    parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-v", "-1" }),
    ExitedWithNonZero(),
    ""
    );
}

/* Zero integer count must exit — selects no features. */
TEST(ParseArgs, VarsCountZeroExits) {
  EXPECT_EXIT(
    parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-v", "0" }),
    ExitedWithNonZero(),
    ""
    );
}

/* Non-numeric string ("abc") must exit. */
TEST(ParseArgs, VarsInvalidValueExits) {
  EXPECT_EXIT(
    parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-v", "abc" }),
    ExitedWithNonZero(),
    ""
    );
}

// ---------------------------------------------------------------------------
// File helpers — ensure_json_extension, check_*_not_exists
// ---------------------------------------------------------------------------

/* Path already ending in .json is returned unchanged. */
TEST(FileHelpers, EnsureJsonExtensionWithExtension) {
  EXPECT_EQ(ensure_json_extension("model.json"), "model.json");
}

/* Path without extension gets .json appended. */
TEST(FileHelpers, EnsureJsonExtensionWithoutExtension) {
  EXPECT_EQ(ensure_json_extension("model"), "model.json");
}

/* Non-.json extension gets .json added (e.g. .txt -> .txt.json). */
TEST(FileHelpers, EnsureJsonExtensionWithOtherExtension) {
  EXPECT_EQ(ensure_json_extension("model.txt"), "model.txt.json");
}

/* Full path without extension gets .json appended. */
TEST(FileHelpers, EnsureJsonExtensionWithPath) {
  EXPECT_EQ(ensure_json_extension("/tmp/model"), "/tmp/model.json");
}

/* check_file_not_exists succeeds for a nonexistent path. */
TEST(FileHelpers, CheckFileNotExistsOnNonexistent) {
  // Should not exit
  check_file_not_exists("/nonexistent/path/that/doesnt/exist.json");
}

/* check_file_not_exists exits for an existing file. */
TEST(FileHelpers, CheckFileNotExistsOnExisting) {
  EXPECT_EXIT(
    check_file_not_exists(IRIS_PATH),
    ExitedWithNonZero(),
    ""
    );
}

/* check_dir_not_exists succeeds for a nonexistent path. */
TEST(FileHelpers, CheckDirNotExistsOnNonexistent) {
  // Should not exit
  check_dir_not_exists("/nonexistent/path/that/doesnt/exist");
}

/* check_dir_not_exists exits for an existing directory. */
TEST(FileHelpers, CheckDirNotExistsOnExisting) {
  EXPECT_EXIT(
    check_dir_not_exists(std::string(PPTREE_DATA_DIR)),
    ExitedWithNonZero(),
    ""
    );
}

// ---------------------------------------------------------------------------
// get_peak_rss_bytes() — memory measurement
// ---------------------------------------------------------------------------

/* The process must report a positive peak RSS value. */
TEST(PeakRSS, ReturnsPositiveValue) {
  long rss = get_peak_rss_bytes();
  EXPECT_GT(rss, 0);
}
