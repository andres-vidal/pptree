#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/vars/Uniform.hpp"
#include "models/strategies/vars/VariableSelection.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::vars;
using namespace ppforest2::stats;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(VarsUniformStrategy, FromJsonValid) {
  json const j  = {{"name", "uniform"}, {"count", 3}};
  auto strategy = Uniform::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(VarsUniformStrategy, FromJsonRoundTrip) {
  json const j  = {{"name", "uniform"}, {"count", 3}};
  auto strategy = Uniform::from_json(j);

  auto out = strategy->to_json();

  EXPECT_EQ(out, j);
}

TEST(VarsUniformStrategy, FromJsonWithProportion) {
  json const j  = {{"name", "uniform"}, {"proportion", 0.5}};
  auto strategy = Uniform::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(VarsUniformStrategy, FromJsonRejectsInvalidProportion) {
  EXPECT_THROW(Uniform::from_json({{"name", "uniform"}, {"proportion", 1.5}}), std::exception);
  EXPECT_THROW(Uniform::from_json({{"name", "uniform"}, {"proportion", 0.0}}), std::exception);
  EXPECT_THROW(Uniform::from_json({{"name", "uniform"}, {"proportion", -0.5}}), std::exception);
}

TEST(VarsUniformStrategy, FromJsonMissingNVars) {
  json const j = {{"name", "uniform"}};
  EXPECT_THROW(Uniform::from_json(j), std::exception);
}

TEST(VarsUniformStrategy, FromJsonUnknownParam) {
  json const j = {{"name", "uniform"}, {"count", 3}, {"extra", true}};
  EXPECT_THROW(Uniform::from_json(j), std::runtime_error);
}

TEST(VarsUniformStrategy, RegistryLookup) {
  json const j  = {{"name", "uniform"}, {"count", 2}};
  auto strategy = VariableSelection::from_json(j);
  ASSERT_NE(strategy, nullptr);

  auto out = strategy->to_json();

  EXPECT_EQ(out, j);
}

TEST(VarsUniformStrategy, SelectsCorrectNumberOfVars) {
  FeatureMatrix const x = MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);

  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1);
  RNG rng(0);

  Uniform const vs(2);
  auto result = vs.compute(x, rng);

  ASSERT_EQ(result.selected_cols.size(), 2U);
  EXPECT_EQ(result.original_size, 5U);
}

TEST(VarsUniformStrategy, AllVarsReturnsAllIndices) {
  FeatureMatrix const x = MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

  RNG rng(0);

  Uniform const vs(3);
  auto result = vs.compute(x, rng);

  ASSERT_EQ(result.selected_cols.size(), 3U);

  std::vector<int> sorted = result.selected_cols;
  std::sort(sorted.begin(), sorted.end());
  EXPECT_EQ(sorted, (std::vector<int>{0, 1, 2}));
}

TEST(VarsUniformStrategy, RejectsZeroVars) {
  EXPECT_THROW(Uniform(0), std::exception);
}

TEST(VarsUniformStrategy, DeterministicWithSameSeed) {
  FeatureMatrix const x = MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);

  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1);
  GroupPartition const gp(y);

  Uniform const vs(2);

  RNG rng1(123);
  auto r1 = vs.compute(x, rng1);

  RNG rng2(123);
  auto r2 = vs.compute(x, rng2);

  EXPECT_EQ(r1.selected_cols, r2.selected_cols);
}
