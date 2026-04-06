#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/stop/PureNode.hpp"
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

TEST(PureNodeStop, FromJsonValid) {
  json const j  = {{"name", "pure_node"}};
  auto strategy = PureNode::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(PureNodeStop, FromJsonRoundTrip) {
  json const j  = {{"name", "pure_node"}};
  auto strategy = PureNode::from_json(j);

  auto out = strategy->to_json();

  EXPECT_EQ(out, j);
}

TEST(PureNodeStop, FromJsonUnknownParam) {
  json const j = {{"name", "pure_node"}, {"extra", 1}};
  EXPECT_THROW(PureNode::from_json(j), std::runtime_error);
}

TEST(PureNodeStop, RegistryLookup) {
  json const j  = {{"name", "pure_node"}};
  auto strategy = StopRule::from_json(j);
  ASSERT_NE(strategy, nullptr);

  auto out = strategy->to_json();
  EXPECT_EQ(out, j);
}

TEST(PureNodeStop, RegistryUnknownStrategy) {
  json const j = {{"name", "unknown_stop"}};
  EXPECT_THROW(StopRule::from_json(j), std::runtime_error);
}

TEST(PureNodeStop, StopsOnSingleGroup) {
  FeatureMatrix const x = MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6);
  OutcomeVector const y = VEC(Outcome, 0, 0, 0);
  GroupPartition const gp(y);
  RNG rng(0);

  NodeContext const ctx(x, gp, 0);
  PureNode const rule;
  EXPECT_TRUE(rule.should_stop(ctx, rng));
}

TEST(PureNodeStop, DoesNotStopOnTwoGroups) {
  FeatureMatrix const x = MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8);
  OutcomeVector const y = VEC(Outcome, 0, 0, 1, 1);
  GroupPartition const gp(y);
  RNG rng(0);

  NodeContext const ctx(x, gp, 0);
  PureNode const rule;
  EXPECT_FALSE(rule.should_stop(ctx, rng));
}

TEST(PureNodeStop, DoesNotStopOnThreeGroups) {
  FeatureMatrix const x = MAT(Feature, rows(6), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
  OutcomeVector const y = VEC(Outcome, 0, 0, 1, 1, 2, 2);
  GroupPartition const gp(y);
  RNG rng(0);

  NodeContext const ctx(x, gp, 0);
  PureNode const rule;
  EXPECT_FALSE(rule.should_stop(ctx, rng));
}

TEST(PureNodeStop, IgnoresDepth) {
  FeatureMatrix const x = MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6);
  OutcomeVector const y = VEC(Outcome, 0, 0, 0);
  GroupPartition const gp(y);
  RNG rng(0);

  PureNode const rule;

  NodeContext const ctx0(x, gp, 0);
  EXPECT_TRUE(rule.should_stop(ctx0, rng));

  NodeContext const ctx10(x, gp, 10);
  EXPECT_TRUE(rule.should_stop(ctx10, rng));

  NodeContext const ctx100(x, gp, 100);
  EXPECT_TRUE(rule.should_stop(ctx100, rng));
}

TEST(PureNodeStop, DisplayName) {
  PureNode const rule;
  EXPECT_EQ(rule.display_name(), "Pure node");
}
