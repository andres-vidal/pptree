#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/partition/ByGroup.hpp"
#include "models/strategies/partition/StepPartition.hpp"
#include "models/strategies/NodeContext.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::partition;
using namespace ppforest2::stats;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(ByGroupPartition, FromJsonValid) {
  json const j  = {{"name", "by_group"}};
  auto strategy = ByGroup::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(ByGroupPartition, FromJsonRoundTrip) {
  json const j  = {{"name", "by_group"}};
  auto strategy = ByGroup::from_json(j);

  auto out = strategy->to_json();

  EXPECT_EQ(out, j);
}

TEST(ByGroupPartition, FromJsonUnknownParam) {
  json const j = {{"name", "by_group"}, {"extra", 0}};
  EXPECT_THROW(ByGroup::from_json(j), std::runtime_error);
}

TEST(ByGroupPartition, RegistryLookup) {
  json const j  = {{"name", "by_group"}};
  auto strategy = StepPartition::from_json(j);
  ASSERT_NE(strategy, nullptr);

  auto out = strategy->to_json();
  EXPECT_EQ(out, j);
}

TEST(ByGroupPartition, RegistryUnknownStrategy) {
  json const j = {{"name", "unknown_partition"}};
  EXPECT_THROW(StepPartition::from_json(j), std::runtime_error);
}

TEST(ByGroupPartition, SplitsBinaryPartition) {
  // 3 original groups remapped to 2 binary groups: {0,1} -> 0, {2} -> 1
  OutcomeVector const y = VEC(Outcome, 0, 0, 1, 1, 2, 2);
  GroupPartition const gp(y);

  std::map<int, int> const mapping = {{0, 0}, {1, 0}, {2, 1}};
  GroupPartition const binary_y    = gp.remap(mapping);

  Outcome const group_0 = 0;
  Outcome const group_1 = 1;

  ByGroup const bg;
  auto const result = bg.compute(binary_y, group_0, group_1);

  EXPECT_EQ(result.lower.groups, (std::set<int>{0, 1}));
  EXPECT_EQ(result.upper.groups, (std::set<int>{2}));
}

TEST(ByGroupPartition, AllRowsAccountedFor) {
  // 4 original groups remapped: {0,1} -> 0, {2,3} -> 1
  FeatureMatrix const x = MAT(Feature, rows(8), 1, 2, 3, 4, 5, 6, 7, 8);
  OutcomeVector const y = VEC(Outcome, 0, 0, 1, 1, 2, 2, 3, 3);
  GroupPartition const gp(y);

  std::map<int, int> const mapping = {{0, 0}, {1, 0}, {2, 1}, {3, 1}};
  GroupPartition const binary_y    = gp.remap(mapping);

  ByGroup const bg;
  auto result     = bg.compute(binary_y, 0, 1);
  auto lower_data = result.lower.data(x).eval();
  auto upper_data = result.upper.data(x).eval();

  EXPECT_EQ_DATA(lower_data, (MAT(Feature, rows(4), 1, 2, 3, 4)));
  EXPECT_EQ_DATA(upper_data, (MAT(Feature, rows(4), 5, 6, 7, 8)));
}

TEST(ByGroupPartition, AllGroupsAccountedFor) {
  OutcomeVector const y = VEC(Outcome, 0, 0, 1, 1, 2, 2, 3, 3);
  GroupPartition const gp(y);

  std::map<int, int> const mapping = {{0, 0}, {1, 0}, {2, 1}, {3, 1}};
  GroupPartition const binary_y    = gp.remap(mapping);

  ByGroup const bg;
  auto result = bg.compute(binary_y, 0, 1);

  std::set<int> u;
  std::set_union(
      result.lower.groups.begin(),
      result.lower.groups.end(),
      result.upper.groups.begin(),
      result.upper.groups.end(),
      std::inserter(u, u.begin())
  );

  EXPECT_EQ(u, gp.groups);
}

TEST(ByGroupPartition, NoOverlap) {
  OutcomeVector const y = VEC(Outcome, 0, 0, 1, 1, 2, 2);
  GroupPartition const gp(y);

  std::map<int, int> const mapping = {{0, 0}, {1, 1}, {2, 1}};
  GroupPartition const binary_y    = gp.remap(mapping);

  ByGroup const bg;
  auto const result = bg.compute(binary_y, 0, 1);


  std::set<int> i;
  std::set_intersection(
      result.lower.groups.begin(),
      result.lower.groups.end(),
      result.upper.groups.begin(),
      result.upper.groups.end(),
      std::inserter(i, i.begin())
  );

  EXPECT_EQ(i, std::set<int>()) << "Groups in lower and upper should not overlap";
}

TEST(ByGroupPartition, NodeContextInterface) {
  // Test the NodeContext-based split() method
  FeatureMatrix const x = MAT(Feature, rows(6), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
  OutcomeVector const y = VEC(Outcome, 0, 0, 1, 1, 2, 2);
  GroupPartition const gp(y);
  RNG rng(0);

  NodeContext ctx(x, gp, 0);

  std::map<int, int> const mapping = {{0, 0}, {1, 0}, {2, 1}};
  ctx.binary_y.emplace(gp.remap(mapping));
  ctx.binary_0 = 0;
  ctx.binary_1 = 1;

  ByGroup const bg;
  auto result = bg.split(ctx, rng);

  EXPECT_EQ(result.lower.groups, (std::set<int>{0, 1}));
  EXPECT_EQ(result.upper.groups, (std::set<int>{2}));
}

TEST(ByGroupPartition, DisplayName) {
  ByGroup const bg;
  EXPECT_EQ(bg.display_name(), "By group");
}
