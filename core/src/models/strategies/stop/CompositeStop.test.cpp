#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/stop/CompositeStop.hpp"
#include "models/strategies/stop/PureNode.hpp"
#include "models/strategies/stop/MinSize.hpp"
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

TEST(CompositeStop, FromJsonRoundTrip) {
  json const j = {
      {"name", "any"}, {"rules", json::array({{{"name", "pure_node"}}, {{"name", "min_size"}, {"min_size", 5}}})}
  };

  auto strategy = CompositeStop::from_json(j);

  auto out = strategy->to_json();

  EXPECT_EQ(out, j);
}

TEST(CompositeStop, FromJsonUnknownParam) {
  json const j = {{"name", "any"}, {"rules", json::array({{{"name", "pure_node"}}})}, {"extra", 1}};
  EXPECT_THROW(CompositeStop::from_json(j), std::runtime_error);
}

TEST(CompositeStop, RegistryLookup) {
  json const j = {
      {"name", "any"}, {"rules", json::array({{{"name", "pure_node"}}, {{"name", "min_size"}, {"min_size", 5}}})}
  };

  auto strategy = StopRule::from_json(j);
  ASSERT_NE(strategy, nullptr);

  auto out = strategy->to_json();
  EXPECT_EQ(out, j);
}

TEST(CompositeStop, DisplayName) {
  auto rule = any({pure_node(), min_size(5)});
  EXPECT_EQ(rule->display_name(), "Any(Pure node, Min size (5))");
}

TEST(CompositeStop, FiresWhenFirstRuleFires) {
  // Pure node on a single-group node → first rule fires
  FeatureMatrix x       = MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6);
  GroupIdVector const y = VEC(GroupId, 0, 0, 0);
  GroupPartition const gp(y);
  OutcomeVector ov = y.cast<Outcome>();
  RNG rng(0);

  NodeContext const ctx(x, gp, ov, 0);
  auto rule = any({pure_node()});
  EXPECT_TRUE(rule->should_stop(ctx, rng));
}

TEST(CompositeStop, FiresWhenSecondRuleFires) {
  // min_size(10) won't fire (12 obs), but pure_node() fires (single group)
  FeatureMatrix x =
      MAT(Feature, rows(12), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
  GroupIdVector const y = VEC(GroupId, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  GroupPartition const gp(y);
  OutcomeVector ov = y.cast<Outcome>();
  RNG rng(0);

  NodeContext const ctx(x, gp, ov, 0);
  auto rule = any({min_size(10), pure_node()});
  // min_size(10): 12 >= 10 → false. pure_node: single group → true.
  EXPECT_TRUE(rule->should_stop(ctx, rng));
}

TEST(CompositeStop, DoesNotFireWhenNoRuleFires) {
  // Two groups → pure_node won't fire
  FeatureMatrix x       = MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1);
  GroupPartition const gp(y);
  OutcomeVector ov = y.cast<Outcome>();
  RNG rng(0);

  NodeContext const ctx(x, gp, ov, 0);
  auto rule = any({pure_node()});
  EXPECT_FALSE(rule->should_stop(ctx, rng));
}

TEST(CompositeStop, RejectsEmptyRules) {
  // An empty composite is a no-op: `should_stop` would never fire and
  // `supported_modes` would claim both modes. Combined with a grouping
  // strategy that can fail to make progress (e.g. ByCutpoint), this yields
  // unbounded recursion. Reject at construction time.
  EXPECT_THROW(any({}), std::exception);
}

TEST(CompositeStop, FromJsonRejectsEmptyRules) {
  nlohmann::json const j = {{"name", "any"}, {"rules", nlohmann::json::array()}};
  EXPECT_THROW(CompositeStop::from_json(j), std::exception);
}
