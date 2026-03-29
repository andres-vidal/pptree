/**
 * @file Benchmark.integration.test.cpp
 * @brief Integration tests for the benchmark subcommand.
 */
#include "cli/CLI.integration.hpp"

// ---------------------------------------------------------------------------
// Benchmark subcommand
// ---------------------------------------------------------------------------

static const std::string MINIMAL_SCENARIOS = R"({
  "name": "integration-test",
  "defaults": {
    "train_ratio": 0.7,
    "seed": 0,
    "lambda": 0.5,
    "iterations": 1
  },
  "scenarios": [
    { "name": "tiny-forest", "n": 50, "p": 3, "g": 2, "size": 5, "vars": 0.5 },
    { "name": "tiny-tree",   "n": 50, "p": 3, "g": 2, "size": 0 }
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
TEST(CLIBenchmark, BenchmarkRunsSuccessfully) {
  auto scenarios = write_scenarios();
  auto result    = run_ppforest2("-q --no-color benchmark -s " + scenarios.path());
  EXPECT_EQ(result.exit_code, 0);
}

/* Benchmark without -s must fail. */
TEST(CLIBenchmark, BenchmarkNoScenariosFails) {
  auto result = run_ppforest2("-q --no-color benchmark");
  EXPECT_NE(result.exit_code, 0);
}

/* Benchmark with invalid scenarios file must fail. */
TEST(CLIBenchmark, BenchmarkInvalidScenariosFails) {
  TempFile bad;
  {
    std::ofstream out(bad.path());
    out << "not valid json";
  }
  auto result = run_ppforest2("-q --no-color benchmark -s " + bad.path());
  EXPECT_NE(result.exit_code, 0);
}

/* Benchmark -o produces valid JSON results. */
TEST(CLIBenchmark, BenchmarkJsonOutput) {
  auto scenarios = write_scenarios();
  TempFile output;
  output.clear();

  auto result = run_ppforest2("-q --no-color benchmark -s " + scenarios.path() + " -o " + output.path());
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
    EXPECT_TRUE(r.contains("size"));
    EXPECT_TRUE(r.contains("mean_time_ms"));
    EXPECT_TRUE(r.contains("mean_test_error"));
  }
}

/* Benchmark --csv produces valid CSV with header and data rows. */
TEST(CLIBenchmark, BenchmarkCsvOutput) {
  auto scenarios = write_scenarios();
  TempFile csv_out(".csv");
  csv_out.clear();

  auto result = run_ppforest2("-q --no-color benchmark -s " + scenarios.path() + " --csv " + csv_out.path());
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
TEST(CLIBenchmark, BenchmarkBaselineComparison) {
  auto scenarios = write_scenarios();

  // First run: produce baseline
  TempFile baseline;
  baseline.clear();
  auto run1 = run_ppforest2("-q --no-color benchmark -s " + scenarios.path() + " -o " + baseline.path());
  ASSERT_EQ(run1.exit_code, 0);

  // Read the baseline to verify it was written and to synchronize the
  // file-system state before the next process checks CLI::ExistingFile.
  ASSERT_FALSE(baseline.read().empty());

  // Second run: compare against baseline
  auto run2 = run_ppforest2("-q --no-color benchmark -s " + scenarios.path() + " -b " + baseline.path());
  EXPECT_EQ(run2.exit_code, 0);
}

/* Benchmark -i overrides iteration count. */
TEST(CLIBenchmark, BenchmarkIterationOverride) {
  auto scenarios = write_scenarios();
  TempFile output;
  output.clear();

  auto result = run_ppforest2("-q --no-color benchmark -s " + scenarios.path() + " -i 2 -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  for (const auto& r : j["results"]) {
    EXPECT_EQ(r["runs"].get<int>(), 2);
  }
}

/* Benchmark with missing scenarios file must fail. */
TEST(CLIBenchmark, BenchmarkMissingFileFails) {
  auto result = run_ppforest2("-q --no-color benchmark -s /nonexistent/path.json");
  EXPECT_NE(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Benchmark with explicit strategy config
// ---------------------------------------------------------------------------

/* Benchmark scenarios with explicit strategy objects run successfully. */
TEST(CLIBenchmark, BenchmarkWithStrategyConfig) {
  static const std::string STRATEGY_SCENARIOS = R"({
    "name": "strategy-test",
    "defaults": {
      "train_ratio": 0.7,
      "seed": 0,
      "iterations": 1,
      "pp": { "name": "pda", "lambda": 0.3 },
      "sr": { "name": "mean_of_means" }
    },
    "scenarios": [
      { "name": "strat-forest", "n": 50, "p": 3, "g": 2, "size": 5,
        "dr": { "name": "uniform", "n_vars": 2 } },
      { "name": "strat-tree",   "n": 50, "p": 3, "g": 2, "size": 0,
        "dr": { "name": "noop" } }
    ]
  })";

  TempFile scenarios;
  {
    std::ofstream out(scenarios.path());
    out << STRATEGY_SCENARIOS;
  }

  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q --no-color benchmark -s " + scenarios.path() + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_EQ(j["results"].size(), 2u);
}
