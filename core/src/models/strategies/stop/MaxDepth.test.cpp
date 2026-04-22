#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/stop/MaxDepth.hpp"
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

TEST(MaxDepthStop, FromJsonValid) {
  json const j  = {{"name", "max_depth"}, {"max_depth", 3}};
  auto strategy = MaxDepth::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(MaxDepthStop, FromJsonRoundTrip) {
  json const j  = {{"name", "max_depth"}, {"max_depth", 3}};
  auto strategy = MaxDepth::from_json(j);
  EXPECT_EQ(strategy->to_json(), j);
}

TEST(MaxDepthStop, FromJsonUnknownParam) {
  json const j = {{"name", "max_depth"}, {"max_depth", 3}, {"extra", 1}};
  EXPECT_THROW(MaxDepth::from_json(j), std::runtime_error);
}

TEST(MaxDepthStop, RegistryLookup) {
  json const j  = {{"name", "max_depth"}, {"max_depth", 3}};
  auto strategy = StopRule::from_json(j);
  ASSERT_NE(strategy, nullptr);
  EXPECT_EQ(strategy->to_json(), j);
}

TEST(MaxDepthStop, DisplayName) {
  MaxDepth const rule(3);
  EXPECT_EQ(rule.display_name(), "Max depth (3)");
}

TEST(MaxDepthStop, StopsWhenDepthReachesMax) {
  // Rule is "stop when ctx.depth >= max_depth" — fires at the boundary.
  FeatureMatrix x       = MAT(Feature, rows(2), 1, 2, 3, 4);
  GroupIdVector const y = VEC(GroupId, 0, 1);
  GroupPartition const gp(y);
  OutcomeVector ov = y.cast<Outcome>();
  RNG rng(0);

  MaxDepth const rule(3);
  EXPECT_TRUE(rule.should_stop(NodeContext(x, gp, ov, 3), rng));
  EXPECT_TRUE(rule.should_stop(NodeContext(x, gp, ov, 4), rng));
  EXPECT_FALSE(rule.should_stop(NodeContext(x, gp, ov, 2), rng));
}

TEST(MaxDepthStop, ZeroMaxDepthProducesStump) {
  // max_depth=0 fires at the root — unusual but explicitly allowed.
  FeatureMatrix x       = MAT(Feature, rows(2), 1, 2, 3, 4);
  GroupIdVector const y = VEC(GroupId, 0, 1);
  GroupPartition const gp(y);
  OutcomeVector ov = y.cast<Outcome>();
  RNG rng(0);

  MaxDepth const rule(0);
  EXPECT_TRUE(rule.should_stop(NodeContext(x, gp, ov, 0), rng));
}

TEST(MaxDepthStop, RejectsNegativeMaxDepth) {
  // A negative max would fire at the root for every tree. The constructor
  // rejects it to prevent silently-degenerate models.
  EXPECT_THROW(MaxDepth(-1), std::exception);
  EXPECT_THROW(MaxDepth(-100), std::exception);
}
