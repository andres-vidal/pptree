/**
 * @file Benchmark.test.cpp
 * @brief Tests for benchmark scenario parsing, validation, and convergence.
 */
#include <gtest/gtest.h>
#include "cli/Benchmark.hpp"

using namespace ppforest2::cli;
using json = nlohmann::json;

class BenchmarkParsingTest : public ::testing::Test {};

TEST_F(BenchmarkParsingTest, ParseMinimalScenario) {
  json j = {
    {
      "scenarios", {
        {
          { "name", "test" },
          { "n", 100 },
          { "p", 5 },
          { "g", 2 }
        }
      }
    }
  };

  auto suite = parse_suite(j);

  ASSERT_EQ(suite.scenarios.size(), 1u);
  EXPECT_EQ(suite.scenarios[0].name, "test");
  EXPECT_EQ(suite.scenarios[0].n, 100);
  EXPECT_EQ(suite.scenarios[0].p, 5);
  EXPECT_EQ(suite.scenarios[0].g, 2);
  // Check defaults
  EXPECT_EQ(suite.scenarios[0].trees, 100);
  EXPECT_FLOAT_EQ(suite.scenarios[0].vars, 0.5f);
  EXPECT_FLOAT_EQ(suite.scenarios[0].lambda, 0.5f);
}

TEST_F(BenchmarkParsingTest, ParseWithDefaults) {
  json j = {
    {
      "defaults", {
        { "trees", 50 },
        { "lambda", 0.3 },
        { "seed", 123 },
        { "warmup", 3 }
      }
    },
    {
      "scenarios", {
        { { "name", "a" }, { "n", 200 }, { "p", 10 }, { "g", 3 } },
        { { "name", "b" }, { "n", 500 }, { "p", 20 }, { "g", 4 }, { "trees", 200 } }
      }
    }
  };

  auto suite = parse_suite(j);

  ASSERT_EQ(suite.scenarios.size(), 2u);

  // Scenario a inherits all defaults
  EXPECT_EQ(suite.scenarios[0].trees, 50);
  EXPECT_FLOAT_EQ(suite.scenarios[0].lambda, 0.3f);
  EXPECT_EQ(suite.scenarios[0].seed, 123);
  EXPECT_EQ(suite.scenarios[0].warmup, 3);

  // Scenario b overrides trees
  EXPECT_EQ(suite.scenarios[1].trees, 200);
  EXPECT_FLOAT_EQ(suite.scenarios[1].lambda, 0.3f);
  EXPECT_EQ(suite.scenarios[1].seed, 123);
}

TEST_F(BenchmarkParsingTest, ParseWithConvergence) {
  json j = {
    {
      "defaults", {
        {
          "convergence", {
            { "cv_threshold", 0.03 },
            { "max_iterations", 100 }
          }
        }
      }
    },
    {
      "scenarios", {
        { { "name", "a" }, { "n", 100 }, { "p", 5 }, { "g", 2 } }
      }
    }
  };

  auto suite = parse_suite(j);

  EXPECT_FLOAT_EQ(suite.scenarios[0].convergence.cv_threshold, 0.03f);
  EXPECT_EQ(suite.scenarios[0].convergence.max_iterations, 100);
  // Default stable_window
  EXPECT_EQ(suite.scenarios[0].convergence.stable_window, 3);
}

TEST_F(BenchmarkParsingTest, FixedIterationsOverrideConvergence) {
  json j = {
    {
      "defaults", {
        { "convergence", { { "max_iterations", 100 } } }
      }
    },
    {
      "scenarios", {
        { { "name", "fixed" }, { "n", 100 }, { "p", 5 }, { "g", 2 }, { "iterations", 5 } }
      }
    }
  };

  auto suite = parse_suite(j);

  EXPECT_EQ(suite.scenarios[0].iterations, 5);
}

TEST_F(BenchmarkParsingTest, ParseSuiteName) {
  json j = {
    {
      "name", "my benchmarks" },
    {
      "scenarios", {
        { { "name", "a" }, { "n", 100 }, { "p", 5 }, { "g", 2 } }
      }
    }
  };

  auto suite = parse_suite(j);

  EXPECT_EQ(suite.name, "my benchmarks");
}

TEST_F(BenchmarkParsingTest, MissingScenariosArrayThrows) {
  json j = { { "defaults", {} } };

  EXPECT_THROW(parse_suite(j), std::runtime_error);
}

TEST_F(BenchmarkParsingTest, EmptyScenariosArrayThrows) {
  json j = { { "scenarios", json::array() } };

  EXPECT_THROW(parse_suite(j), std::runtime_error);
}

TEST_F(BenchmarkParsingTest, InvalidScenarioThrows) {
  // g must be > 1
  json j = {
    {
      "scenarios", {
        { { "name", "bad" }, { "n", 100 }, { "p", 5 }, { "g", 1 } }
      }
    }
  };

  EXPECT_THROW(parse_suite(j), std::runtime_error);
}

TEST_F(BenchmarkParsingTest, MissingNameThrows) {
  json j = {
    {
      "scenarios", {
        { { "n", 100 }, { "p", 5 }, { "g", 2 } }
      }
    }
  };

  EXPECT_THROW(parse_suite(j), std::runtime_error);
}

TEST_F(BenchmarkParsingTest, NegativeNThrows) {
  json j = {
    {
      "scenarios", {
        { { "name", "bad" }, { "n", -1 }, { "p", 5 }, { "g", 2 } }
      }
    }
  };

  EXPECT_THROW(parse_suite(j), std::runtime_error);
}

TEST_F(BenchmarkParsingTest, ParseDefaultScenariosFile) {
  auto suite = parse_suite(std::string(PPFOREST2_BENCH_SCENARIOS));

  EXPECT_GT(suite.scenarios.size(), 0u);

  for (const auto& s : suite.scenarios) {
    EXPECT_FALSE(s.name.empty());

    if (s.data.empty()) {
      // Simulation scenarios must have valid dimensions
      EXPECT_GT(s.n, 0);
      EXPECT_GT(s.p, 0);
      EXPECT_GT(s.g, 1);
    } else {
      // Data scenarios must have a non-empty path
      EXPECT_FALSE(s.data.empty());
    }
  }
}

class SuiteResultTest : public ::testing::Test {};

TEST_F(SuiteResultTest, ToJsonRoundtrip) {
  SuiteResult result;
  result.suite_name    = "test suite";
  result.timestamp     = "2026-01-01T00:00:00";
  result.total_time_ms = 1234.5;

  ScenarioResult sr;
  sr.name           = "scenario1";
  sr.n              = 100;
  sr.p              = 5;
  sr.g              = 2;
  sr.trees          = 50;
  sr.vars           = 0.5f;
  sr.runs           = 10;
  sr.mean_time_ms   = 12.3;
  sr.std_time_ms    = 1.2;
  sr.mean_tr_error  = 0.01;
  sr.mean_te_error  = 0.05;
  sr.peak_rss_bytes = 1024 * 1024;
  sr.peak_rss_mb    = 1.0;

  result.results.push_back(sr);

  auto j = result.to_json();

  EXPECT_EQ(j["suite_name"], "test suite");
  EXPECT_EQ(j["timestamp"], "2026-01-01T00:00:00");
  EXPECT_DOUBLE_EQ(j["total_time_ms"], 1234.5);

  ASSERT_EQ(j["results"].size(), 1u);
  EXPECT_EQ(j["results"][0]["name"], "scenario1");
  EXPECT_EQ(j["results"][0]["n"], 100);
  EXPECT_EQ(j["results"][0]["trees"], 50);
  EXPECT_DOUBLE_EQ(j["results"][0]["mean_time_ms"], 12.3);
  EXPECT_EQ(j["results"][0]["peak_rss_bytes"], 1024 * 1024);
}

TEST_F(SuiteResultTest, ToJsonOmitsPeakRSSWhenNegative) {
  SuiteResult result;
  result.suite_name = "test";

  ScenarioResult sr;
  sr.name           = "no-rss";
  sr.peak_rss_bytes = -1;
  sr.peak_rss_mb    = -1;

  result.results.push_back(sr);

  auto j = result.to_json();

  EXPECT_FALSE(j["results"][0].contains("peak_rss_bytes"));
  EXPECT_FALSE(j["results"][0].contains("peak_rss_mb"));
}

TEST_F(BenchmarkParsingTest, ParseDataScenario) {
  json j = {
    {
      "scenarios", {
        { { "name", "csv-test" }, { "data", "data/iris.csv" }, { "trees", 50 } }
      }
    }
  };

  auto suite = parse_suite(j);

  ASSERT_EQ(suite.scenarios.size(), 1u);
  EXPECT_EQ(suite.scenarios[0].name, "csv-test");
  EXPECT_EQ(suite.scenarios[0].data, "data/iris.csv");
  EXPECT_EQ(suite.scenarios[0].trees, 50);
}

TEST_F(BenchmarkParsingTest, DataScenarioSkipsNPGValidation) {
  // A data scenario with default n/p/g (which would be fine anyway)
  // but the point is that validation doesn't require explicit n/p/g
  json j = {
    {
      "scenarios", {
        { { "name", "csv-test" }, { "data", "some/file.csv" } }
      }
    }
  };

  EXPECT_NO_THROW(parse_suite(j));
}

TEST_F(BenchmarkParsingTest, DataScenarioToJsonIncludesDataField) {
  SuiteResult result;
  result.suite_name = "test";

  ScenarioResult sr;
  sr.name = "csv-test";
  sr.data = "data/iris.csv";
  sr.n    = 150;
  sr.p    = 4;
  sr.g    = 3;

  result.results.push_back(sr);

  auto j = result.to_json();

  EXPECT_EQ(j["results"][0]["data"], "data/iris.csv");
  EXPECT_EQ(j["results"][0]["n"], 150);
}

TEST_F(BenchmarkParsingTest, SimulationScenarioToJsonOmitsDataField) {
  SuiteResult result;
  result.suite_name = "test";

  ScenarioResult sr;
  sr.name = "sim-test";
  sr.n    = 100;
  sr.p    = 5;
  sr.g    = 2;

  result.results.push_back(sr);

  auto j = result.to_json();

  EXPECT_FALSE(j["results"][0].contains("data"));
}

TEST_F(BenchmarkParsingTest, ScenarioVarsAsIntegerCount) {
  json j = {
    {
      "scenarios", {
        { { "name", "int-vars" }, { "n", 100 }, { "p", 10 }, { "g", 2 }, { "vars", 3 } }
      }
    }
  };

  auto suite = parse_suite(j);

  EXPECT_FLOAT_EQ(suite.scenarios[0].vars, 0.3f);
}

TEST_F(BenchmarkParsingTest, ScenarioVarsAsFraction) {
  json j = {
    {
      "scenarios", {
        { { "name", "frac-vars" }, { "n", 100 }, { "p", 10 }, { "g", 2 }, { "vars", "1/3" } }
      }
    }
  };

  auto suite = parse_suite(j);

  EXPECT_NEAR(suite.scenarios[0].vars, 1.0f / 3.0f, 1e-6);
}
