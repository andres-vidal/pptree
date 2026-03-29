#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/DRUniformStrategy.hpp"
#include "models/DRStrategy.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::dr;
using namespace ppforest2::stats;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(DRUniformStrategy, FromJsonValid) {
  json j        = { { "name", "uniform" }, { "n_vars", 3 } };
  auto strategy = DRUniformStrategy::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(DRUniformStrategy, FromJsonRoundTrip) {
  json j        = { { "name", "uniform" }, { "n_vars", 3 } };
  auto strategy = DRUniformStrategy::from_json(j);

  json out;
  strategy->to_json(out);

  EXPECT_EQ(out["name"], "uniform");
  EXPECT_EQ(out["n_vars"], 3);
}

TEST(DRUniformStrategy, FromJsonMissingNVars) {
  json j = { { "name", "uniform" } };
  EXPECT_THROW(DRUniformStrategy::from_json(j), std::exception);
}

TEST(DRUniformStrategy, FromJsonUnknownParam) {
  json j = { { "name", "uniform" }, { "n_vars", 3 }, { "extra", true } };
  EXPECT_THROW(DRUniformStrategy::from_json(j), std::runtime_error);
}

TEST(DRUniformStrategy, RegistryLookup) {
  json j        = { { "name", "uniform" }, { "n_vars", 2 } };
  auto strategy = DRStrategy::from_json(j);
  ASSERT_NE(strategy, nullptr);

  json out;
  strategy->to_json(out);
  EXPECT_EQ(out["name"], "uniform");
  EXPECT_EQ(out["n_vars"], 2);
}

TEST(DRUniformStrategy, SelectsCorrectNumberOfVars) {
  FeatureMatrix x = MAT(Feature, rows(4),
      1, 2, 3, 4, 5,
      6, 7, 8, 9, 10,
      11, 12, 13, 14, 15,
      16, 17, 18, 19, 20);

  ResponseVector y = VEC(Response, 0, 0, 1, 1);
  GroupPartition gp(y);
  RNG rng(0);

  DRUniformStrategy dr(2);
  auto result = dr.select(x, gp, rng);

  EXPECT_EQ(result.selected_cols.size(), 2u);
  EXPECT_EQ(result.original_size, 5);
}

TEST(DRUniformStrategy, AllVarsReturnsAllIndices) {
  FeatureMatrix x = MAT(Feature, rows(4),
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,
      10, 11, 12);

  ResponseVector y = VEC(Response, 0, 0, 1, 1);
  GroupPartition gp(y);
  RNG rng(0);

  DRUniformStrategy dr(3);
  auto result = dr.select(x, gp, rng);

  ASSERT_EQ(result.selected_cols.size(), 3u);

  std::vector<int> sorted = result.selected_cols;
  std::sort(sorted.begin(), sorted.end());
  EXPECT_EQ(sorted, (std::vector<int>{ 0, 1, 2 }));
}

TEST(DRUniformStrategy, RejectsZeroVars) {
  EXPECT_THROW(DRUniformStrategy(0), std::exception);
}

TEST(DRUniformStrategy, DeterministicWithSameSeed) {
  FeatureMatrix x = MAT(Feature, rows(4),
      1, 2, 3, 4, 5,
      6, 7, 8, 9, 10,
      11, 12, 13, 14, 15,
      16, 17, 18, 19, 20);

  ResponseVector y = VEC(Response, 0, 0, 1, 1);
  GroupPartition gp(y);

  DRUniformStrategy dr(2);

  RNG rng1(123);
  auto r1 = dr.select(x, gp, rng1);

  RNG rng2(123);
  auto r2 = dr.select(x, gp, rng2);

  EXPECT_EQ(r1.selected_cols, r2.selected_cols);
}
