#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/stop/MinVariance.hpp"
#include "models/strategies/stop/StopRule.hpp"
#include "models/strategies/NodeContext.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::stop;
using namespace ppforest2::stats;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(MinVarianceStop, FromJsonValid) {
  json const j  = {{"name", "min_variance"}, {"threshold", 0.01f}};
  auto strategy = MinVariance::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(MinVarianceStop, FromJsonRoundTrip) {
  json const j  = {{"name", "min_variance"}, {"threshold", 0.01f}};
  auto strategy = MinVariance::from_json(j);

  auto out = strategy->to_json();

  EXPECT_EQ(out, j);
}

TEST(MinVarianceStop, FromJsonUnknownParam) {
  json const j = {{"name", "min_variance"}, {"threshold", 0.01f}, {"extra", 1}};
  EXPECT_THROW(MinVariance::from_json(j), std::runtime_error);
}

TEST(MinVarianceStop, RegistryLookup) {
  json const j  = {{"name", "min_variance"}, {"threshold", 0.01f}};
  auto strategy = StopRule::from_json(j);
  ASSERT_NE(strategy, nullptr);

  auto out = strategy->to_json();
  EXPECT_EQ(out, j);
}

TEST(MinVarianceStop, DisplayName) {
  // Default-float formatting renders small thresholds compactly and falls
  // back to scientific when fixed-notation would round them to zero.
  EXPECT_EQ(MinVariance(0.01f).display_name(), "Min variance (0.01)");
  EXPECT_EQ(MinVariance(1e-6f).display_name(), "Min variance (1e-06)");
}

TEST(MinVarianceStop, StopsWhenVarianceIsZero) {
  FeatureMatrix x       = MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6);
  GroupIdVector const y = VEC(GroupId, 0, 0, 0);
  GroupPartition const gp(y);
  OutcomeVector ov = VEC(Feature, 1.0, 1.0, 1.0);
  RNG rng(0);

  NodeContext ctx(x, gp, ov, 0);

  MinVariance const rule(0.01f);
  EXPECT_TRUE(rule.should_stop(ctx, rng));
}

TEST(MinVarianceStop, DoesNotStopWhenVarianceIsHigh) {
  FeatureMatrix x       = MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6);
  GroupIdVector const y = VEC(GroupId, 0, 0, 0);
  GroupPartition const gp(y);
  OutcomeVector ov = VEC(Feature, 1.0, 10.0, 20.0);
  RNG rng(0);

  NodeContext ctx(x, gp, ov, 0);

  MinVariance const rule(0.01f);
  EXPECT_FALSE(rule.should_stop(ctx, rng));
}

TEST(MinVarianceStop, StopsWithSingleObservation) {
  FeatureMatrix x       = MAT(Feature, rows(1), 1, 2);
  GroupIdVector const y = VEC(GroupId, 0);
  GroupPartition const gp(y);
  OutcomeVector ov = VEC(Feature, 5.0);
  RNG rng(0);

  NodeContext ctx(x, gp, ov, 0);

  MinVariance const rule(0.01f);
  // Single observation → variance undefined → should stop
  EXPECT_TRUE(rule.should_stop(ctx, rng));
}
