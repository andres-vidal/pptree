#include <gtest/gtest.h>

#include "pptree.hpp"
#include "DataPacket.hpp"
#include "CLIOptions.hpp"
#include "IO.hpp"

using namespace pptree;

static const std::string IRIS_PATH = std::string(PPTREE_DATA_DIR) + "/iris.csv";

// Death test predicate: exits with any non-zero code
class ExitedWithNonZero {
  public:
    bool operator()(int exit_status) const {
#ifdef _WIN32
      return exit_status != 0;

#else
      return testing::ExitedWithCode(0)(exit_status) == false
             && WIFEXITED(exit_status);

#endif
    }
};

// Helper to build argv and call parse_args
static CLIOptions parse(std::initializer_list<const char *> args_list) {
  std::vector<const char *> args(args_list);
  return parse_args(static_cast<int>(args.size()), const_cast<char **>(args.data()));
}

// ---------------------------------------------------------------------------
// parse_args — Subcommand Routing
// ---------------------------------------------------------------------------

TEST(ParseArgs, TrainSubcommand) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str() });
  EXPECT_EQ(opts.subcommand, Subcommand::train);
}

TEST(ParseArgs, EvaluateSubcommandWithSimulate) {
  auto opts = parse({ "pptree", "evaluate", "-s", "100x5x2" });
  EXPECT_EQ(opts.subcommand, Subcommand::evaluate);
}

TEST(ParseArgs, EvaluateSubcommandWithData) {
  auto opts = parse({ "pptree", "evaluate", "-d", IRIS_PATH.c_str() });
  EXPECT_EQ(opts.subcommand, Subcommand::evaluate);
  EXPECT_EQ(opts.data_path, IRIS_PATH);
}

TEST(ParseArgs, NoSubcommandExits) {
  EXPECT_EXIT(
    parse({ "pptree" }),
    ExitedWithNonZero(),
    ""
    );
}

TEST(ParseArgs, InvalidSubcommandExits) {
  EXPECT_EXIT(
    parse({ "pptree", "foobar" }),
    ExitedWithNonZero(),
    ""
    );
}

// ---------------------------------------------------------------------------
// parse_args — Train Options
// ---------------------------------------------------------------------------

TEST(ParseArgs, TrainDefaultValues) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str() });
  EXPECT_EQ(opts.trees, 100);
  EXPECT_FLOAT_EQ(opts.lambda, 0.5f);
  EXPECT_EQ(opts.threads, -1);
  EXPECT_EQ(opts.seed, -1);
  EXPECT_FLOAT_EQ(opts.p_vars, 0.5f);
  EXPECT_EQ(opts.n_vars, -1);
  EXPECT_TRUE(opts.save_path.empty());
}

TEST(ParseArgs, TrainTreesOption) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-t", "50" });
  EXPECT_EQ(opts.trees, 50);
}

TEST(ParseArgs, TrainLambdaOption) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-l", "0.3" });
  EXPECT_FLOAT_EQ(opts.lambda, 0.3f);
}

TEST(ParseArgs, TrainThreadsOption) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-n", "4" });
  EXPECT_EQ(opts.threads, 4);
}

TEST(ParseArgs, TrainSeedOption) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-r", "42" });
  EXPECT_EQ(opts.seed, 42);
}

TEST(ParseArgs, TrainPVarsOption) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-v", "0.8" });
  EXPECT_FLOAT_EQ(opts.p_vars, 0.8f);
}

TEST(ParseArgs, TrainNVarsOption) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-m", "3" });
  EXPECT_EQ(opts.n_vars, 3);
}

TEST(ParseArgs, TrainSaveOption) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-o", "/tmp/model.json" });
  EXPECT_EQ(opts.save_path, "/tmp/model.json");
}

TEST(ParseArgs, TrainZeroTrees) {
  auto opts = parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-t", "0" });
  EXPECT_EQ(opts.trees, 0);
}

TEST(ParseArgs, TrainMissingDataExits) {
  EXPECT_EXIT(
    parse({ "pptree", "train" }),
    ExitedWithNonZero(),
    ""
    );
}

TEST(ParseArgs, TrainNonexistentDataExits) {
  EXPECT_EXIT(
    parse({ "pptree", "train", "-d", "/nonexistent/file.csv" }),
    ExitedWithNonZero(),
    ""
    );
}

TEST(ParseArgs, TrainLambdaOutOfRangeExits) {
  EXPECT_EXIT(
    parse({ "pptree", "train", "-d", IRIS_PATH.c_str(), "-l", "2.0" }),
    ExitedWithNonZero(),
    ""
    );
}

// ---------------------------------------------------------------------------
// parse_args — Evaluate Options
// ---------------------------------------------------------------------------

TEST(ParseArgs, EvaluateSimulateFormat) {
  auto opts = parse({ "pptree", "evaluate", "-s", "100x5x2" });
  EXPECT_EQ(opts.rows, 100);
  EXPECT_EQ(opts.cols, 5);
  EXPECT_EQ(opts.classes, 2);
}

TEST(ParseArgs, EvaluateSimulateCustomParams) {
  auto opts = parse({ "pptree", "evaluate", "-s", "100x5x3",
                      "--sim-mean", "200", "--sim-mean-separation", "25", "--sim-sd", "5" });
  EXPECT_FLOAT_EQ(opts.sim_mean, 200.0f);
  EXPECT_FLOAT_EQ(opts.sim_mean_separation, 25.0f);
  EXPECT_FLOAT_EQ(opts.sim_sd, 5.0f);
}

TEST(ParseArgs, EvaluateTrainRatio) {
  auto opts = parse({ "pptree", "evaluate", "-s", "100x5x2", "-p", "0.8" });
  EXPECT_FLOAT_EQ(opts.train_ratio, 0.8f);
}

TEST(ParseArgs, EvaluateNRuns) {
  auto opts = parse({ "pptree", "evaluate", "-s", "100x5x2", "-e", "5" });
  EXPECT_EQ(opts.n_runs, 5);
}

TEST(ParseArgs, EvaluateInvalidSimulateFormatExits) {
  EXPECT_EXIT(
    parse({ "pptree", "evaluate", "-s", "100x5" }),
    ExitedWithNonZero(),
    ""
    );
}

TEST(ParseArgs, EvaluateClassesMustBeGreaterThanOne) {
  EXPECT_EXIT(
    parse({ "pptree", "evaluate", "-s", "100x5x1" }),
    ExitedWithNonZero(),
    ""
    );
}

TEST(ParseArgs, EvaluateNoDataSourceExits) {
  EXPECT_EXIT(
    parse({ "pptree", "evaluate" }),
    ExitedWithNonZero(),
    ""
    );
}

TEST(ParseArgs, EvaluateSimParamsNeedSimulate) {
  EXPECT_EXIT(
    parse({ "pptree", "evaluate", "-d", IRIS_PATH.c_str(), "--sim-mean", "200" }),
    ExitedWithNonZero(),
    ""
    );
}

// ---------------------------------------------------------------------------
// parse_args — Predict Options
// ---------------------------------------------------------------------------

TEST(ParseArgs, PredictMissingModelExits) {
  EXPECT_EXIT(
    parse({ "pptree", "predict", "-d", IRIS_PATH.c_str() }),
    ExitedWithNonZero(),
    ""
    );
}

TEST(ParseArgs, PredictMissingDataExits) {
  EXPECT_EXIT(
    parse({ "pptree", "predict", "-M", IRIS_PATH.c_str() }),
    ExitedWithNonZero(),
    ""
    );
}

// ---------------------------------------------------------------------------
// parse_args — Global Options
// ---------------------------------------------------------------------------

TEST(ParseArgs, OutputFormatJson) {
  auto opts = parse({ "pptree", "--output-format=json", "evaluate", "-s", "100x5x2" });
  EXPECT_EQ(opts.output_format, OutputFormat::json);
  EXPECT_TRUE(opts.quiet);
}

TEST(ParseArgs, OutputFormatText) {
  auto opts = parse({ "pptree", "--output-format=text", "evaluate", "-s", "100x5x2" });
  EXPECT_EQ(opts.output_format, OutputFormat::text);
}

TEST(ParseArgs, OutputFormatInvalidExits) {
  EXPECT_EXIT(
    parse({ "pptree", "--output-format=xml", "evaluate", "-s", "100x5x2" }),
    ExitedWithNonZero(),
    ""
    );
}

TEST(ParseArgs, QuietShortFlag) {
  auto opts = parse({ "pptree", "-q", "evaluate", "-s", "100x5x2" });
  EXPECT_TRUE(opts.quiet);
}

TEST(ParseArgs, QuietLongFlag) {
  auto opts = parse({ "pptree", "--quiet", "evaluate", "-s", "100x5x2" });
  EXPECT_TRUE(opts.quiet);
}

// ---------------------------------------------------------------------------
// init_params
// ---------------------------------------------------------------------------

TEST(InitParams, DefaultLambda) {
  CLIOptions params;
  params.lambda = -1;
  params.quiet  = true;
  init_params(params);
  EXPECT_FLOAT_EQ(params.lambda, 0.5f);
}

TEST(InitParams, LambdaUnchangedIfSet) {
  CLIOptions params;
  params.lambda = 0.3f;
  params.quiet  = true;
  init_params(params);
  EXPECT_FLOAT_EQ(params.lambda, 0.3f);
}

TEST(InitParams, InvalidTrainRatioZeroExits) {
  EXPECT_EXIT({
    CLIOptions params;
    params.train_ratio = 0;
    params.quiet       = true;
    init_params(params);
  }, ExitedWithNonZero(), "");
}

TEST(InitParams, InvalidTrainRatioOneExits) {
  EXPECT_EXIT({
    CLIOptions params;
    params.train_ratio = 1.0f;
    params.quiet       = true;
    init_params(params);
  }, ExitedWithNonZero(), "");
}

TEST(InitParams, InvalidTrainRatioNegativeExits) {
  EXPECT_EXIT({
    CLIOptions params;
    params.train_ratio = -0.5f;
    params.quiet       = true;
    init_params(params);
  }, ExitedWithNonZero(), "");
}

TEST(InitParams, AutoSeed) {
  CLIOptions params;
  params.seed  = -1;
  params.quiet = true;
  init_params(params);
  EXPECT_NE(params.seed, -1);
}

TEST(InitParams, SeedPreservedIfSet) {
  CLIOptions params;
  params.seed  = 42;
  params.quiet = true;
  init_params(params);
  EXPECT_EQ(params.seed, 42);
}

TEST(InitParams, DefaultThreads) {
  CLIOptions params;
  params.threads = -1;
  params.quiet   = true;
  init_params(params);
  EXPECT_GE(params.threads, 1);
}

TEST(InitParams, ThreadsPreservedIfSet) {
  CLIOptions params;
  params.threads = 8;
  params.quiet   = true;
  init_params(params);
  EXPECT_EQ(params.threads, 8);
}

TEST(InitParams, NVarsFromPVars) {
  CLIOptions params;
  params.trees  = 10;
  params.p_vars = 0.5f;
  params.n_vars = -1;
  params.quiet  = true;
  init_params(params, 10);
  EXPECT_EQ(params.n_vars, 5);
}

TEST(InitParams, PVarsFromNVars) {
  CLIOptions params;
  params.trees  = 10;
  params.p_vars = -1;
  params.n_vars = 3;
  params.quiet  = true;
  init_params(params, 10);
  EXPECT_FLOAT_EQ(params.p_vars, 0.3f);
}

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
// warn_unused_params
// ---------------------------------------------------------------------------

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

TEST(WarnUnusedParams, TreesZeroPVarsWarning) {
  CLIOptions params;
  params.trees  = 0;
  params.p_vars = 0.8f;
  params.quiet  = false;

  testing::internal::CaptureStdout();
  warn_unused_params(params);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("var-proportion parameter is ignored"), std::string::npos);
}

TEST(WarnUnusedParams, TreesZeroBothWarnings) {
  CLIOptions params;
  params.trees   = 0;
  params.threads = 4;
  params.p_vars  = 0.8f;
  params.quiet   = false;

  testing::internal::CaptureStdout();
  warn_unused_params(params);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("threads parameter is ignored"), std::string::npos);
  EXPECT_NE(output.find("var-proportion parameter is ignored"), std::string::npos);
  EXPECT_NE(output.find("Single trees always use all features"), std::string::npos);
}

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
// ModelStats
// ---------------------------------------------------------------------------

TEST(ModelStats, MeanTime) {
  ModelStats stats;
  stats.tr_times = models::stats::DataColumn<double>(3);
  stats.tr_times << 10.0, 20.0, 30.0;

  EXPECT_DOUBLE_EQ(stats.mean_time(), 20.0);
}

TEST(ModelStats, MeanTrainError) {
  ModelStats stats;
  stats.tr_error = models::stats::DataColumn<double>(3);
  stats.tr_error << 0.1, 0.2, 0.3;

  EXPECT_DOUBLE_EQ(stats.mean_tr_error(), 0.2);
}

TEST(ModelStats, MeanTestError) {
  ModelStats stats;
  stats.te_error = models::stats::DataColumn<double>(3);
  stats.te_error << 0.05, 0.15, 0.25;

  EXPECT_DOUBLE_EQ(stats.mean_te_error(), 0.15);
}

TEST(ModelStats, ToJson) {
  ModelStats stats;
  stats.tr_times = models::stats::DataColumn<double>(2);
  stats.tr_times << 10.0, 20.0;
  stats.tr_error = models::stats::DataColumn<double>(2);
  stats.tr_error << 0.1, 0.3;
  stats.te_error = models::stats::DataColumn<double>(2);
  stats.te_error << 0.2, 0.4;

  auto j = stats.to_json();

  EXPECT_EQ(j["runs"], 2);
  EXPECT_DOUBLE_EQ(j["mean_time_ms"].get<double>(), 15.0);
  EXPECT_DOUBLE_EQ(j["mean_train_error"].get<double>(), 0.2);
  EXPECT_DOUBLE_EQ(j["mean_test_error"].get<double>(), 0.3);
}
