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

#include "ppforest2.hpp"
#include "cli/CLIOptions.hpp"
#include "io/Color.hpp"
#include "io/Output.hpp"

#include <filesystem>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::cli;
using ppforest2::io::Output;


#ifndef PPFOREST2_DATA_DIR
#error "PPFOREST2_DATA_DIR must be defined"
#endif

static const std::string IRIS_PATH = std::string(PPFOREST2_DATA_DIR) + "/iris.csv";

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
 *   auto opts = parse({ "ppforest2", "train", "-d", "data.csv" });
 *
 * @param args_list Initializer list of C-string arguments (first must be program name).
 * @return Populated CLIOptions struct.
 */
static CLIOptions parse(std::initializer_list<char const*> args_list) {
  std::vector<char const*> args(args_list);
  return parse_args(static_cast<int>(args.size()), const_cast<char**>(args.data()));
}

// ---------------------------------------------------------------------------
// parse_args() — subcommand routing
// ---------------------------------------------------------------------------

/* Parsing "train" selects the train subcommand. */
TEST(ParseArgs, TrainSubcommand) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str()});
  EXPECT_EQ(opts.subcommand, Subcommand::train);
}

/* Parsing "evaluate" with --simulate selects the evaluate subcommand. */
TEST(ParseArgs, EvaluateSubcommandWithSimulate) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2"});
  EXPECT_EQ(opts.subcommand, Subcommand::evaluate);
}

/* Parsing "evaluate" with -d selects evaluate and captures the data path. */
TEST(ParseArgs, EvaluateSubcommandWithData) {
  auto opts = parse({"ppforest2", "evaluate", "-d", IRIS_PATH.c_str()});
  EXPECT_EQ(opts.subcommand, Subcommand::evaluate);
  EXPECT_EQ(opts.data_path, IRIS_PATH);
}

/* Omitting a subcommand must exit with a non-zero code. */
TEST(ParseArgs, NoSubcommandExits) {
  EXPECT_EXIT(parse({"ppforest2"}), ExitedWithNonZero(), "");
}

/* An unrecognised subcommand must exit with a non-zero code. */
TEST(ParseArgs, InvalidSubcommandExits) {
  EXPECT_EXIT(parse({"ppforest2", "foobar"}), ExitedWithNonZero(), "");
}

// ---------------------------------------------------------------------------
// parse_args() — train defaults and options
// ---------------------------------------------------------------------------

/* Verify all default values when only -d is supplied. */
TEST(ParseArgs, TrainDefaultValues) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str()});
  EXPECT_EQ(opts.model.size, 100);
  EXPECT_FLOAT_EQ(opts.model.lambda, 0.5f);
  EXPECT_EQ(opts.model.threads, -1);
  EXPECT_EQ(opts.model.seed, -1);
  EXPECT_FLOAT_EQ(opts.model.p_vars, -1);
  EXPECT_EQ(opts.model.n_vars, -1);
  EXPECT_TRUE(opts.model.vars_input.empty());
  EXPECT_EQ(opts.save_path, "model.json");
  EXPECT_FALSE(opts.no_save);
}

/* -n overrides the default tree count. */
TEST(ParseArgs, TrainSizeOption) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-n", "50"});
  EXPECT_EQ(opts.model.size, 50);
}

/* -l overrides the default lambda. */
TEST(ParseArgs, TrainLambdaOption) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-l", "0.3"});
  EXPECT_FLOAT_EQ(opts.model.lambda, 0.3f);
}

/* --threads sets the thread count explicitly. */
TEST(ParseArgs, TrainThreadsOption) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--threads", "4"});
  EXPECT_EQ(opts.model.threads, 4);
}

/* -r sets the random seed. */
TEST(ParseArgs, TrainSeedOption) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-r", "0"});
  EXPECT_EQ(opts.model.seed, 0);
}

// ---------------------------------------------------------------------------
// parse_args() — --n-vars and --p-vars (feature sub-sampling) parsing
// ---------------------------------------------------------------------------

/* --p-vars with a decimal value (0.8) is interpreted as a proportion. */
TEST(ParseArgs, PVarsAsProportion) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "0.8"});
  EXPECT_FLOAT_EQ(opts.model.p_vars, 0.8F);
  EXPECT_EQ(opts.model.n_vars, -1);
}

/* --n-vars with an integer (3) is interpreted as an absolute feature count. */
TEST(ParseArgs, NVarsAsAbsoluteCount) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--n-vars", "3"});
  EXPECT_EQ(opts.model.n_vars, 3);
  EXPECT_FLOAT_EQ(opts.model.p_vars, -1);
}

/* --p-vars with a fraction "1/3" is parsed as a proportion. */
TEST(ParseArgs, PVarsAsFraction) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "1/3"});
  EXPECT_NEAR(opts.model.p_vars, 1.0F / 3.0F, 0.001F);
  EXPECT_EQ(opts.model.n_vars, -1);
}

/* --p-vars "1/2" parses to exactly 0.5. */
TEST(ParseArgs, PVarsAsFractionHalf) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "1/2"});
  EXPECT_FLOAT_EQ(opts.model.p_vars, 0.5F);
  EXPECT_EQ(opts.model.n_vars, -1);
}

/* --p-vars "3/3" parses to 1.0 (use all features). */
TEST(ParseArgs, PVarsAsFractionFull) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "3/3"});
  EXPECT_FLOAT_EQ(opts.model.p_vars, 1.0F);
  EXPECT_EQ(opts.model.n_vars, -1);
}

/* --p-vars with fraction > 1 (4/3) must exit. */
TEST(ParseArgs, PVarsFractionGreaterThanOneExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "4/3"}), ExitedWithNonZero(), "");
}

/* --p-vars with division by zero (1/0) must exit. */
TEST(ParseArgs, PVarsFractionZeroDenominatorExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "1/0"}), ExitedWithNonZero(), "");
}

/* --p-vars with zero numerator (0/3) must exit — no features selected. */
TEST(ParseArgs, PVarsFractionZeroNumeratorExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "0/3"}), ExitedWithNonZero(), "");
}

/* --p-vars with non-numeric fraction ("a/b") must exit. */
TEST(ParseArgs, PVarsFractionInvalidExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "a/b"}), ExitedWithNonZero(), "");
}

/* --n-vars and --p-vars are mutually exclusive. */
TEST(ParseArgs, NVarsExcludesPVars) {
  EXPECT_EXIT(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--n-vars", "3", "--p-vars", "0.5"}),
      ExitedWithNonZero(),
      ""
  );
}

// ---------------------------------------------------------------------------
// parse_args() — save, no-save, and lambda validation
// ---------------------------------------------------------------------------

/* -s sets a custom model output path. */
TEST(ParseArgs, TrainSaveOption) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-s", "/tmp/model.json"});
  EXPECT_EQ(opts.save_path, "/tmp/model.json");
}

/* Zero trees is accepted (means a single projection-pursuit tree). */
TEST(ParseArgs, TrainZeroTrees) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-n", "0"});
  EXPECT_EQ(opts.model.size, 0);
}

/* Train without -d must exit. */
TEST(ParseArgs, TrainMissingDataExits) {
  EXPECT_EXIT(parse({"ppforest2", "train"}), ExitedWithNonZero(), "");
}

/* Train with a nonexistent data file must exit. */
TEST(ParseArgs, TrainNonexistentDataExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", "/nonexistent/file.csv"}), ExitedWithNonZero(), "");
}

/* Lambda = 0 is a valid boundary value (pure LDA). */
TEST(ParseArgs, TrainLambdaZeroValid) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-l", "0"});
  EXPECT_FLOAT_EQ(opts.model.lambda, 0.0F);
}

/* Lambda = 1 is a valid boundary value (pure PCA). */
TEST(ParseArgs, TrainLambdaOneValid) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-l", "1.0"});
  EXPECT_FLOAT_EQ(opts.model.lambda, 1.0F);
}

/* Negative lambda must exit. */
TEST(ParseArgs, TrainLambdaNegativeExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-l", "-0.1"}), ExitedWithNonZero(), "");
}

/* Lambda > 1 must exit. */
TEST(ParseArgs, TrainLambdaOutOfRangeExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-l", "2.0"}), ExitedWithNonZero(), "");
}

/* --no-save disables model saving and clears the save path. */
TEST(ParseArgs, TrainNoSaveFlag) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--no-save"});
  EXPECT_TRUE(opts.no_save);
  EXPECT_TRUE(opts.save_path.empty());
}

/* --no-save and -s together must exit (mutually exclusive). */
TEST(ParseArgs, TrainNoSaveExcludesSave) {
  EXPECT_EXIT(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--no-save", "-s", "/tmp/m.json"}), ExitedWithNonZero(), ""
  );
}

// ---------------------------------------------------------------------------
// parse_args() — predict options
// ---------------------------------------------------------------------------

/* -o sets the prediction output file path. */
TEST(ParseArgs, PredictOutputOption) {
  auto opts =
      parse({"ppforest2", "predict", "-M", IRIS_PATH.c_str(), "-d", IRIS_PATH.c_str(), "-o", "/tmp/results.json"});
  EXPECT_EQ(opts.output_path, "/tmp/results.json");
}

/* --no-metrics flag is captured. */
TEST(ParseArgs, PredictNoMetricsFlag) {
  auto opts = parse({"ppforest2", "predict", "-M", IRIS_PATH.c_str(), "-d", IRIS_PATH.c_str(), "--no-metrics"});
  EXPECT_TRUE(opts.no_metrics);
}

/* --no-proportions flag is captured. */
TEST(ParseArgs, PredictNoProportionsFlag) {
  auto opts = parse({"ppforest2", "predict", "-M", IRIS_PATH.c_str(), "-d", IRIS_PATH.c_str(), "--no-proportions"});
  EXPECT_TRUE(opts.no_proportions);
}

/* --no-proportions defaults to false (proportions included by default). */
TEST(ParseArgs, PredictNoProportionsDefault) {
  auto opts = parse({"ppforest2", "predict", "-M", IRIS_PATH.c_str(), "-d", IRIS_PATH.c_str()});
  EXPECT_FALSE(opts.no_proportions);
}

/* Predict without -M (model path) must exit. */
TEST(ParseArgs, PredictMissingModelExits) {
  EXPECT_EXIT(parse({"ppforest2", "predict", "-d", IRIS_PATH.c_str()}), ExitedWithNonZero(), "");
}

/* Predict without -d (data path) must exit. */
TEST(ParseArgs, PredictMissingDataExits) {
  EXPECT_EXIT(parse({"ppforest2", "predict", "-M", IRIS_PATH.c_str()}), ExitedWithNonZero(), "");
}

// ---------------------------------------------------------------------------
// parse_args() — evaluate options
// ---------------------------------------------------------------------------

/* -o sets the evaluation output file path. */
TEST(ParseArgs, EvaluateOutputOption) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "-o", "/tmp/results.json"});
  EXPECT_EQ(opts.output_path, "/tmp/results.json");
}

/* Evaluate must reject -s (--save is train-only). */
TEST(ParseArgs, EvaluateNoSaveOption) {
  // Evaluate should not accept -s
  EXPECT_EXIT(
      parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "-s", "/tmp/model.json"}), ExitedWithNonZero(), ""
  );
}

/* Evaluate always has an empty save_path (save is not available). */
TEST(ParseArgs, EvaluateSavePathAlwaysEmpty) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2"});
  EXPECT_TRUE(opts.save_path.empty());
}

/* -e sets the experiment export directory. */
TEST(ParseArgs, EvaluateExportOption) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "-e", "/tmp/experiment"});
  EXPECT_EQ(opts.evaluate.export_path, "/tmp/experiment");
}

/* --simulate "RxCxK" is parsed into rows, cols, and groups. */
TEST(ParseArgs, EvaluateSimulateFormat) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2"});
  EXPECT_EQ(opts.simulation.rows, 100);
  EXPECT_EQ(opts.simulation.cols, 5);
  EXPECT_EQ(opts.simulation.n_groups, 2);
}

/* Simulation generation parameters (mean, separation, sd) are captured. */
TEST(ParseArgs, EvaluateSimulateCustomParams) {
  auto opts = parse(
      {"ppforest2",
       "evaluate",
       "--simulate",
       "100x5x3",
       "--simulate-mean",
       "200",
       "--simulate-mean-separation",
       "25",
       "--simulate-sd",
       "5"}
  );
  EXPECT_FLOAT_EQ(opts.simulation.mean, 200.0F);
  EXPECT_FLOAT_EQ(opts.simulation.mean_separation, 25.0F);
  EXPECT_FLOAT_EQ(opts.simulation.sd, 5.0F);
}

/* -p sets the train/test split ratio. */
TEST(ParseArgs, EvaluateTrainRatio) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "-p", "0.8"});
  EXPECT_FLOAT_EQ(opts.evaluate.train_ratio, 0.8F);
}

/* -i sets fixed iteration count and disables convergence. */
TEST(ParseArgs, EvaluateIterations) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "-i", "5"});
  EXPECT_EQ(opts.evaluate.iterations, 5);
  EXPECT_FALSE(opts.convergence.enabled);
}

/* Malformed --simulate string (missing dimension) must exit. */
TEST(ParseArgs, EvaluateInvalidSimulateFormatExits) {
  EXPECT_EXIT(parse({"ppforest2", "evaluate", "--simulate", "100x5"}), ExitedWithNonZero(), "");
}

/* Simulation with only 1 group must exit (need >= 2 for classification). */
TEST(ParseArgs, EvaluateGroupsMustBeGreaterThanOne) {
  EXPECT_EXIT(parse({"ppforest2", "evaluate", "--simulate", "100x5x1"}), ExitedWithNonZero(), "");
}

/* Evaluate without -d or --simulate must exit. */
TEST(ParseArgs, EvaluateNoDataSourceExits) {
  EXPECT_EXIT(parse({"ppforest2", "evaluate"}), ExitedWithNonZero(), "");
}

/* Providing both -d and --simulate must exit (mutually exclusive). */
TEST(ParseArgs, EvaluateBothDataAndSimulateExits) {
  EXPECT_EXIT(
      parse({"ppforest2", "evaluate", "-d", IRIS_PATH.c_str(), "--simulate", "100x5x2"}), ExitedWithNonZero(), ""
  );
}

// ---------------------------------------------------------------------------
// parse_args() — convergence defaults and -i override
// ---------------------------------------------------------------------------

/* Evaluate defaults to convergence mode. */
TEST(ParseArgs, EvaluateDefaultConvergence) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2"});
  EXPECT_TRUE(opts.convergence.enabled);
  EXPECT_EQ(opts.convergence.max, 200);
  EXPECT_FLOAT_EQ(opts.convergence.cv, 0.05F);
  EXPECT_EQ(opts.convergence.min, 10);
  EXPECT_EQ(opts.convergence.window, 3);
}

/* -i disables convergence and sets fixed iteration count. */
TEST(ParseArgs, IterationsDisablesConvergence) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "-i", "5"});
  EXPECT_FALSE(opts.convergence.enabled);
  EXPECT_EQ(opts.evaluate.iterations, 5);
}

/* --convergence-max overrides convergence cap without disabling it. */
TEST(ParseArgs, MaxIterationsOverride) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "--convergence-max", "500"});
  EXPECT_TRUE(opts.convergence.enabled);
  EXPECT_EQ(opts.convergence.max, 500);
}

/* --convergence-cv overrides the convergence threshold. */
TEST(ParseArgs, CvThresholdOverride) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "--convergence-cv", "0.01"});
  EXPECT_TRUE(opts.convergence.enabled);
  EXPECT_FLOAT_EQ(opts.convergence.cv, 0.01F);
}

/* --convergence-min overrides the minimum before convergence checking. */
TEST(ParseArgs, MinIterationsOverride) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "--convergence-min", "20"});
  EXPECT_TRUE(opts.convergence.enabled);
  EXPECT_EQ(opts.convergence.min, 20);
}

/* --convergence-window overrides the consecutive stable iterations required. */
TEST(ParseArgs, StableWindowOverride) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "--convergence-window", "5"});
  EXPECT_TRUE(opts.convergence.enabled);
  EXPECT_EQ(opts.convergence.window, 5);
}

/* All convergence parameters can be set together. */
TEST(ParseArgs, AllConvergenceParams) {
  auto opts = parse(
      {"ppforest2",
       "evaluate",
       "--simulate",
       "100x5x2",
       "--convergence-cv",
       "0.02",
       "--convergence-max",
       "300",
       "--convergence-min",
       "15",
       "--convergence-window",
       "4"}
  );
  EXPECT_TRUE(opts.convergence.enabled);
  EXPECT_FLOAT_EQ(opts.convergence.cv, 0.02F);
  EXPECT_EQ(opts.convergence.max, 300);
  EXPECT_EQ(opts.convergence.min, 15);
  EXPECT_EQ(opts.convergence.window, 4);
}

/* -i takes precedence: convergence params are parsed but converge is false. */
TEST(ParseArgs, IterationsOverridesConvergenceParams) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "-i", "10", "--convergence-cv", "0.01"});
  EXPECT_FALSE(opts.convergence.enabled);
  EXPECT_EQ(opts.evaluate.iterations, 10);
  // --convergence-cv is still parsed but converge is disabled
  EXPECT_FLOAT_EQ(opts.convergence.cv, 0.01F);
}

/* Simulation parameters without --simulate must exit. */
TEST(ParseArgs, EvaluateSimParamsNeedSimulate) {
  EXPECT_EXIT(
      parse({"ppforest2", "evaluate", "-d", IRIS_PATH.c_str(), "--simulate-mean", "200"}), ExitedWithNonZero(), ""
  );
}

// ---------------------------------------------------------------------------
// parse_args() — global flags
// ---------------------------------------------------------------------------

/* --no-color flag is captured when placed before the subcommand. */
TEST(ParseArgs, NoColorFlag) {
  auto opts = parse({"ppforest2", "--no-color", "evaluate", "--simulate", "100x5x2"});
  EXPECT_TRUE(opts.no_color);
}

/* --no-color defaults to false. */
TEST(ParseArgs, NoColorDefaultFalse) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2"});
  EXPECT_FALSE(opts.no_color);
}

/* -q enables quiet mode. */
TEST(ParseArgs, QuietShortFlag) {
  auto opts = parse({"ppforest2", "-q", "evaluate", "--simulate", "100x5x2"});
  EXPECT_TRUE(opts.quiet);
}

/* --quiet also enables quiet mode. */
TEST(ParseArgs, QuietLongFlag) {
  auto opts = parse({"ppforest2", "--quiet", "evaluate", "--simulate", "100x5x2"});
  EXPECT_TRUE(opts.quiet);
}


/* --version prints version info and exits with code 0. */
TEST(ParseArgs, VersionExits) {
  EXPECT_EXIT(parse({"ppforest2", "--version"}), testing::ExitedWithCode(0), "");
}

// ---------------------------------------------------------------------------
// init_params() — default resolution and validation
// ---------------------------------------------------------------------------

/* Sentinel lambda (-1) is replaced by the default 0.5. */
TEST(InitParams, DefaultLambda) {
  CLIOptions params;
  params.model.lambda = -1;
  params.quiet        = true;
  init_params(params);
  EXPECT_FLOAT_EQ(params.model.lambda, 0.5F);
}

/* An explicitly set lambda is preserved. */
TEST(InitParams, LambdaUnchangedIfSet) {
  CLIOptions params;
  params.model.lambda = 0.3F;
  params.quiet        = true;
  init_params(params);
  EXPECT_FLOAT_EQ(params.model.lambda, 0.3F);
}

/* Train ratio of 0 must exit (no training data). */
TEST(InitParams, InvalidTrainRatioZeroExits) {
  EXPECT_EXIT(
      {
        CLIOptions params;
        params.evaluate.train_ratio = 0;
        params.quiet                = true;
        init_params(params);
      },
      ExitedWithNonZero(),
      ""
  );
}

/* Train ratio of 1.0 must exit (no test data). */
TEST(InitParams, InvalidTrainRatioOneExits) {
  EXPECT_EXIT(
      {
        CLIOptions params;
        params.evaluate.train_ratio = 1.0F;
        params.quiet                = true;
        init_params(params);
      },
      ExitedWithNonZero(),
      ""
  );
}

/* Negative train ratio must exit. */
TEST(InitParams, InvalidTrainRatioNegativeExits) {
  EXPECT_EXIT(
      {
        CLIOptions params;
        params.evaluate.train_ratio = -0.5F;
        params.quiet                = true;
        init_params(params);
      },
      ExitedWithNonZero(),
      ""
  );
}

// ---------------------------------------------------------------------------
// init_params() — used_default_* tracking flags
// ---------------------------------------------------------------------------

/* Sentinel seed (-1) sets used_default_seed to true. */
TEST(InitParams, UsedDefaultSeedFlag) {
  CLIOptions params;
  params.model.seed = -1;
  params.quiet      = true;
  init_params(params);
  EXPECT_TRUE(params.model.used_default_seed);
}

/* An explicit seed clears used_default_seed. */
TEST(InitParams, UsedDefaultSeedFlagFalse) {
  CLIOptions params;
  params.model.seed = 0;
  params.quiet      = true;
  init_params(params);
  EXPECT_FALSE(params.model.used_default_seed);
}

/* Sentinel threads (-1) sets used_default_threads to true. */
TEST(InitParams, UsedDefaultThreadsFlag) {
  CLIOptions params;
  params.model.threads = -1;
  params.quiet         = true;
  init_params(params);
  EXPECT_TRUE(params.model.used_default_threads);
}

/* An explicit thread count clears used_default_threads. */
TEST(InitParams, UsedDefaultThreadsFlagFalse) {
  CLIOptions params;
  params.model.threads = 8;
  params.quiet         = true;
  init_params(params);
  EXPECT_FALSE(params.model.used_default_threads);
}

/* Sentinel vars (-1/-1) sets used_default_vars to true. */
TEST(InitParams, UsedDefaultVarsFlag) {
  CLIOptions params;
  params.model.size   = 10;
  params.model.p_vars = -1;
  params.model.n_vars = -1;
  params.quiet        = true;
  init_params(params, 10);
  EXPECT_TRUE(params.model.used_default_vars);
}

/* An explicit p_vars clears used_default_vars. */
TEST(InitParams, UsedDefaultVarsFlagFalse) {
  CLIOptions params;
  params.model.size   = 10;
  params.model.p_vars = 0.8F;
  params.model.n_vars = -1;
  params.quiet        = true;
  init_params(params, 10);
  EXPECT_FALSE(params.model.used_default_vars);
}

// ---------------------------------------------------------------------------
// init_params() — seed, threads, and vars auto-resolution
// ---------------------------------------------------------------------------

/* Sentinel seed (-1) is replaced by a generated seed. */
TEST(InitParams, AutoSeed) {
  CLIOptions params;
  params.model.seed = -1;
  params.quiet      = true;
  init_params(params);
  EXPECT_NE(params.model.seed, -1);
}

/* An explicit seed value is preserved. */
TEST(InitParams, SeedPreservedIfSet) {
  CLIOptions params;
  params.model.seed = 0;
  params.quiet      = true;
  init_params(params);
  EXPECT_EQ(params.model.seed, 0);
}

/* Sentinel threads (-1) auto-detects to >= 1 thread. */
TEST(InitParams, DefaultThreads) {
  CLIOptions params;
  params.model.threads = -1;
  params.quiet         = true;
  init_params(params);
  EXPECT_GE(params.model.threads, 1);
}

/* An explicitly set thread count is preserved. */
TEST(InitParams, ThreadsPreservedIfSet) {
  CLIOptions params;
  params.model.threads = 8;
  params.quiet         = true;
  init_params(params);
  EXPECT_EQ(params.model.threads, 8);
}

/* n_vars is computed from p_vars * total_vars. */
TEST(InitParams, NVarsFromPVars) {
  CLIOptions params;
  params.model.size   = 10;
  params.model.p_vars = 0.5F;
  params.model.n_vars = -1;
  params.quiet        = true;
  init_params(params, 10);
  EXPECT_EQ(params.model.n_vars, 5);
}

/* p_vars is back-computed from n_vars / total_vars. */
TEST(InitParams, PVarsFromNVars) {
  CLIOptions params;
  params.model.size   = 10;
  params.model.p_vars = -1;
  params.model.n_vars = 3;
  params.quiet        = true;
  init_params(params, 10);
  EXPECT_FLOAT_EQ(params.model.p_vars, 0.3F);
}

/* Default vars: p_vars = 0.5, n_vars = half of total. */
TEST(InitParams, DefaultPVarsAndNVars) {
  CLIOptions params;
  params.model.size   = 10;
  params.model.p_vars = -1;
  params.model.n_vars = -1;
  params.quiet        = true;
  init_params(params, 10);
  EXPECT_FLOAT_EQ(params.model.p_vars, 0.5F);
  EXPECT_EQ(params.model.n_vars, 5);
}

/* Vars computation is skipped for a single tree (trees = 0). */
TEST(InitParams, NoVarsWhenSingleTree) {
  CLIOptions params;
  params.model.size   = 0;
  params.model.p_vars = 0.8F;
  params.model.n_vars = -1;
  params.quiet        = true;
  init_params(params, 10);
  EXPECT_FLOAT_EQ(params.model.p_vars, 0.8F);
  EXPECT_EQ(params.model.n_vars, -1);
}

/* Vars computation is skipped when total_vars is 0 (unknown). */
TEST(InitParams, NoVarsWhenZeroTotalVars) {
  CLIOptions params;
  params.model.size   = 10;
  params.model.p_vars = -1;
  params.model.n_vars = -1;
  params.quiet        = true;
  init_params(params, 0);
  EXPECT_FLOAT_EQ(params.model.p_vars, -1);
  EXPECT_EQ(params.model.n_vars, -1);
}

// ---------------------------------------------------------------------------
// warn_unused_params() — single-tree parameter warnings
// ---------------------------------------------------------------------------

/* Single tree with --threads warns that threads is ignored. */
TEST(WarnUnusedParams, TreesZeroThreadsWarning) {
  CLIOptions params;
  params.model.size    = 0;
  params.model.threads = 4;
  params.quiet         = false;

  testing::internal::CaptureStdout();
  ppforest2::io::Output out(params.quiet);
  warn_unused_params(out, params);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("threads parameter is ignored"), std::string::npos);
}

/* Single tree with --vars warns that vars is ignored. */
TEST(WarnUnusedParams, TreesZeroVarsWarning) {
  CLIOptions params;
  params.model.size   = 0;
  params.model.p_vars = 0.8F;
  params.quiet        = false;

  testing::internal::CaptureStdout();
  ppforest2::io::Output out(params.quiet);
  warn_unused_params(out, params);
  std::string const output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("--n-vars/--p-vars parameter is ignored"), std::string::npos);
}

/* Single tree with both --threads and --vars emits both warnings. */
TEST(WarnUnusedParams, TreesZeroBothWarnings) {
  CLIOptions params;
  params.model.size    = 0;
  params.model.threads = 4;
  params.model.n_vars  = 3;
  params.quiet         = false;

  testing::internal::CaptureStdout();
  ppforest2::io::Output out(params.quiet);
  warn_unused_params(out, params);
  std::string const output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("threads parameter is ignored"), std::string::npos);
  EXPECT_NE(output.find("--n-vars/--p-vars parameter is ignored"), std::string::npos);
  EXPECT_NE(output.find("Single trees always use all features"), std::string::npos);
}

/* Forest mode (trees > 0) emits no warnings. */
TEST(WarnUnusedParams, TreesNonZeroNoWarning) {
  CLIOptions params;
  params.model.size    = 10;
  params.model.threads = 4;
  params.model.p_vars  = 0.8F;
  params.quiet         = false;

  testing::internal::CaptureStdout();
  ppforest2::io::Output out(params.quiet);
  warn_unused_params(out, params);
  std::string const output = testing::internal::GetCapturedStdout();

  EXPECT_TRUE(output.empty());
}

/* Quiet mode suppresses all parameter warnings. */
TEST(WarnUnusedParams, QuietSuppresses) {
  CLIOptions params;
  params.model.size    = 0;
  params.model.threads = 4;
  params.model.p_vars  = 0.8f;
  params.quiet         = true;

  testing::internal::CaptureStdout();
  ppforest2::io::Output out(params.quiet);
  warn_unused_params(out, params);
  std::string const output = testing::internal::GetCapturedStdout();

  EXPECT_TRUE(output.empty());
}


// ---------------------------------------------------------------------------
// parse_args() — additional vars validation (edge cases)
// ---------------------------------------------------------------------------

/* --p-vars of exactly 0.0 must exit — selects no features. */
TEST(ParseArgs, PVarsProportionZeroExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "0.0"}), ExitedWithNonZero(), "");
}

/* --p-vars > 1.0 must exit. */
TEST(ParseArgs, PVarsProportionOutOfRangeExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "1.5"}), ExitedWithNonZero(), "");
}

/* --n-vars with negative count must exit. */
TEST(ParseArgs, NVarsNegativeCountExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--n-vars", "-1"}), ExitedWithNonZero(), "");
}

/* --n-vars with zero count must exit — selects no features. */
TEST(ParseArgs, NVarsCountZeroExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--n-vars", "0"}), ExitedWithNonZero(), "");
}

/* --p-vars with non-numeric string ("abc") must exit. */
TEST(ParseArgs, PVarsInvalidValueExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "abc"}), ExitedWithNonZero(), "");
}

// ---------------------------------------------------------------------------
// parse_args() — explicit strategy flags (--pp, --vars, --cutpoint)
// ---------------------------------------------------------------------------

/* --pp pda without lambda must exit — explicit API requires all params. */
TEST(ParseArgs, PPStrategyNameOnlyExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "pda"}), ExitedWithNonZero(), "");
}

/* --pp pda:lambda=0.3 sets lambda to 0.3. */
TEST(ParseArgs, PPStrategyWithLambda) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "pda:lambda=0.3"});
  EXPECT_EQ(opts.model.pp_input, "pda:lambda=0.3");
  EXPECT_FLOAT_EQ(opts.model.pp_config["lambda"].get<float>(), 0.3F);
}

/* --pp and -l are mutually exclusive. */
TEST(ParseArgs, PPExcludesLambda) {
  EXPECT_EXIT(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "pda", "-l", "0.5"}), ExitedWithNonZero(), ""
  );
}

/* -l and --pp are mutually exclusive (reverse order). */
TEST(ParseArgs, LambdaExcludesPP) {
  EXPECT_EXIT(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-l", "0.5", "--pp", "pda"}), ExitedWithNonZero(), ""
  );
}

/* --pp with an unknown strategy name must exit. */
TEST(ParseArgs, PPUnknownStrategyExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "unknown"}), ExitedWithNonZero(), "");
}

/* --pp pda:unknown=1 with an unknown parameter must exit. */
TEST(ParseArgs, PPUnknownParamExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "pda:unknown=1"}), ExitedWithNonZero(), "");
}

/* --pp pda:lambda=notanumber must exit. */
TEST(ParseArgs, PPInvalidLambdaValueExits) {
  EXPECT_EXIT(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "pda:lambda=abc"}), ExitedWithNonZero(), ""
  );
}

/* --pp pda:noequalssign must exit (missing key=value). */
TEST(ParseArgs, PPMissingEqualsExits) {
  EXPECT_EXIT(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "pda:noequalssign"}), ExitedWithNonZero(), ""
  );
}

/* --vars all sets vars to all (forces n_vars=0). */
TEST(ParseArgs, VarsStrategyAll) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--vars", "all"});
  EXPECT_EQ(opts.model.vars_input, "all");
  EXPECT_EQ(opts.model.n_vars, 0);
}

/* --vars uniform:count=2 sets vars for later resolution. */
TEST(ParseArgs, VarsStrategyUniformWithCount) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--vars", "uniform:count=2"});
  EXPECT_EQ(opts.model.n_vars, 2);
}

/* --vars and --n-vars are mutually exclusive. */
TEST(ParseArgs, VarsStrategyExcludesNVars) {
  EXPECT_EXIT(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--vars", "all", "--n-vars", "2"}), ExitedWithNonZero(), ""
  );
}

/* --n-vars and --vars are mutually exclusive (reverse order). */
TEST(ParseArgs, NVarsExcludesVarsStrategy) {
  EXPECT_EXIT(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--n-vars", "2", "--vars", "all"}), ExitedWithNonZero(), ""
  );
}

/* --vars and --p-vars are mutually exclusive. */
TEST(ParseArgs, VarsStrategyExcludesPVars) {
  EXPECT_EXIT(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--vars", "all", "--p-vars", "0.5"}),
      ExitedWithNonZero(),
      ""
  );
}

/* --vars with an unknown strategy name must exit. */
TEST(ParseArgs, VarsStrategyUnknownExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--vars", "unknown"}), ExitedWithNonZero(), "");
}

/* --vars uniform:unknown=1 with an unknown parameter must exit. */
TEST(ParseArgs, VarsStrategyUnknownParamExits) {
  EXPECT_EXIT(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--vars", "uniform:unknown=1"}), ExitedWithNonZero(), ""
  );
}

/* --cutpoint mean_of_means is accepted. */
TEST(ParseArgs, ThresholdStrategyMeanOfMeans) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--cutpoint", "mean_of_means"});
  EXPECT_EQ(opts.model.cutpoint_input, "mean_of_means");
}

/* --cutpoint with an unknown strategy name must exit. */
TEST(ParseArgs, ThresholdUnknownStrategyExits) {
  EXPECT_EXIT(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--cutpoint", "unknown"}), ExitedWithNonZero(), "");
}

/* --max-retries sets the max retries count. */
TEST(ParseArgs, MaxRetriesOption) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--max-retries", "5"});
  EXPECT_EQ(opts.model.max_retries, 5);
}

/* --max-retries defaults to 3. */
TEST(ParseArgs, MaxRetriesDefault) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str()});
  EXPECT_EQ(opts.model.max_retries, 3);
}

/* All three explicit strategy flags can be used together. */
TEST(ParseArgs, AllExplicitStrategies) {
  auto opts = parse(
      {"ppforest2",
       "train",
       "-d",
       IRIS_PATH.c_str(),
       "--pp",
       "pda:lambda=0.3",
       "--vars",
       "uniform:count=2",
       "--cutpoint",
       "mean_of_means"}
  );
  EXPECT_EQ(opts.model.pp_input, "pda:lambda=0.3");
  EXPECT_EQ(opts.model.vars_input, "uniform:count=2");
  EXPECT_EQ(opts.model.cutpoint_input, "mean_of_means");
  EXPECT_FLOAT_EQ(opts.model.pp_config["lambda"].get<float>(), 0.3F);
  EXPECT_EQ(opts.model.vars_config["count"].get<int>(), 2);
}

/* --pp pda:lambda=0.3 correctly parses single param (baseline for multi-param). */
TEST(ParseArgs, PPStrategySingleParamParsed) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "pda:lambda=0.3"});
  EXPECT_EQ(opts.model.pp_config["name"], "pda");
  EXPECT_FLOAT_EQ(opts.model.pp_config["lambda"].get<float>(), 0.3F);
  EXPECT_EQ(opts.model.pp_config.size(), 2u); // name + lambda only
}

/* --pp pda:lambda=0.3,unknown=1 with multiple params rejects unknown. */
TEST(ParseArgs, PPStrategyMultipleParamsRejectsUnknown) {
  EXPECT_EXIT(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "pda:lambda=0.3,unknown=1"}),
      ExitedWithNonZero(),
      ""
  );
}

/* --vars uniform:count=2,unknown=1 with multiple params rejects unknown. */
TEST(ParseArgs, VarsStrategyMultipleParamsRejectsUnknown) {
  EXPECT_EXIT(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--vars", "uniform:count=2,unknown=1"}),
      ExitedWithNonZero(),
      ""
  );
}

// ---------------------------------------------------------------------------
// summarize subcommand parsing
// ---------------------------------------------------------------------------

/* Parsing "summarize" selects the summarize subcommand. */
TEST(ParseArgs, SummarizeSubcommand) {
  auto opts = parse({"ppforest2", "summarize", "-M", IRIS_PATH.c_str()});
  EXPECT_EQ(opts.subcommand, Subcommand::summarize);
  EXPECT_EQ(opts.model_path, IRIS_PATH);
}

/* Summarize without required -M must exit. */
TEST(ParseArgs, SummarizeWithoutModelExits) {
  EXPECT_EXIT(parse({"ppforest2", "summarize"}), ExitedWithNonZero(), "");
}
