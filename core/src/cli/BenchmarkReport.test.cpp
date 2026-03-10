/**
 * @file BenchmarkReport.test.cpp
 * @brief Tests for benchmark report CSV/JSON/markdown export.
 */
#include <gtest/gtest.h>
#include "cli/Benchmark.hpp"
#include "cli/BenchmarkReport.hpp"

#include <fstream>
#include <filesystem>
#include <sstream>

using namespace pptree::cli;

class BenchmarkReportTest : public ::testing::Test {
  protected:
    SuiteResult make_sample_result() {
      SuiteResult result;
      result.suite_name    = "test suite";
      result.timestamp     = "2026-01-01T00:00:00";
      result.total_time_ms = 5000;

      ScenarioResult sr1;
      sr1.name          = "small";
      sr1.n             = 100;
      sr1.p             = 5;
      sr1.g             = 2;
      sr1.trees         = 50;
      sr1.vars          = 0.5f;
      sr1.runs          = 10;
      sr1.mean_time_ms  = 12.3;
      sr1.std_time_ms   = 1.2;
      sr1.mean_tr_error = 0.01;
      sr1.mean_te_error = 0.05;
      sr1.peak_rss_bytes = 5 * 1024 * 1024;
      sr1.peak_rss_mb   = 5.0;
      sr1.scenario_time_ms = 2500;

      ScenarioResult sr2;
      sr2.name          = "large";
      sr2.n             = 5000;
      sr2.p             = 50;
      sr2.g             = 5;
      sr2.trees         = 200;
      sr2.vars          = 0.5f;
      sr2.runs          = 5;
      sr2.mean_time_ms  = 500.0;
      sr2.std_time_ms   = 25.0;
      sr2.mean_tr_error = 0.02;
      sr2.mean_te_error = 0.10;
      sr2.peak_rss_bytes = 100 * 1024 * 1024;
      sr2.peak_rss_mb   = 100.0;
      sr2.scenario_time_ms = 2500;

      result.results.push_back(sr1);
      result.results.push_back(sr2);

      return result;
    }
};

TEST_F(BenchmarkReportTest, WriteAndReadJson) {
  auto result = make_sample_result();
  std::string path = (std::filesystem::temp_directory_path() / "pptree-test-report.json").string();

  write_results_json(result, path);

  auto loaded = parse_results(path);

  EXPECT_EQ(loaded.suite_name, "test suite");
  EXPECT_EQ(loaded.timestamp, "2026-01-01T00:00:00");
  ASSERT_EQ(loaded.results.size(), 2u);
  EXPECT_EQ(loaded.results[0].name, "small");
  EXPECT_EQ(loaded.results[1].name, "large");
  EXPECT_DOUBLE_EQ(loaded.results[0].mean_time_ms, 12.3);
  EXPECT_DOUBLE_EQ(loaded.results[1].mean_time_ms, 500.0);

  std::filesystem::remove(path);
}

TEST_F(BenchmarkReportTest, WriteCsv) {
  auto result = make_sample_result();
  std::string path = (std::filesystem::temp_directory_path() / "pptree-test-report.csv").string();

  write_results_csv(result, path);

  std::ifstream file(path);
  ASSERT_TRUE(file.is_open());

  std::string header;
  std::getline(file, header);
  EXPECT_TRUE(header.find("scenario") != std::string::npos);
  EXPECT_TRUE(header.find("mean_time_ms") != std::string::npos);
  EXPECT_TRUE(header.find("peak_rss_mb") != std::string::npos);

  std::string line1;
  std::getline(file, line1);
  EXPECT_TRUE(line1.find("small") != std::string::npos);

  std::string line2;
  std::getline(file, line2);
  EXPECT_TRUE(line2.find("large") != std::string::npos);

  // No more data lines
  std::string line3;
  std::getline(file, line3);
  EXPECT_TRUE(line3.empty());

  file.close();
  std::filesystem::remove(path);
}

TEST_F(BenchmarkReportTest, FormatMarkdownWithoutBaseline) {
  auto result = make_sample_result();
  auto md = format_benchmark_markdown(result);

  // Header
  EXPECT_TRUE(md.find("## test suite") != std::string::npos);
  EXPECT_TRUE(md.find("2026-01-01T00:00:00") != std::string::npos);

  // Table header (no delta columns)
  EXPECT_TRUE(md.find("| Scenario |") != std::string::npos);
  EXPECT_TRUE(md.find("| Time (ms) |") != std::string::npos);
  EXPECT_TRUE(md.find("| Peak RSS |") != std::string::npos);

  // No delta columns
  std::string delta = "\xCE\x94";
  EXPECT_TRUE(md.find(delta) == std::string::npos);

  // Scenario rows
  EXPECT_TRUE(md.find("| small |") != std::string::npos);
  EXPECT_TRUE(md.find("| large |") != std::string::npos);

  // Footer
  EXPECT_TRUE(md.find("2 scenarios") != std::string::npos);
}

TEST_F(BenchmarkReportTest, FormatMarkdownWithBaseline) {
  auto current = make_sample_result();

  // Baseline with different timing to produce deltas
  auto baseline = make_sample_result();
  baseline.timestamp = "2026-01-01T00:00:00-baseline";
  baseline.results[0].mean_time_ms = 15.0;   // current is 12.3 → improvement
  baseline.results[1].mean_time_ms = 400.0;  // current is 500.0 → regression

  auto md = format_benchmark_markdown(current, baseline);

  // Both timestamps
  EXPECT_TRUE(md.find("Current:") != std::string::npos);
  EXPECT_TRUE(md.find("Baseline:") != std::string::npos);

  // Delta columns present
  std::string delta = "\xCE\x94";
  EXPECT_TRUE(md.find(delta + " Time") != std::string::npos);
  EXPECT_TRUE(md.find(delta + " RSS") != std::string::npos);

  // Green circle for improvement (small scenario: 12.3 vs 15.0 = -18%)
  std::string green = "\xF0\x9F\x9F\xA2";
  EXPECT_TRUE(md.find(green) != std::string::npos);

  // Red circle for regression (large scenario: 500 vs 400 = +25%)
  std::string red = "\xF0\x9F\x94\xB4";
  EXPECT_TRUE(md.find(red) != std::string::npos);
}
