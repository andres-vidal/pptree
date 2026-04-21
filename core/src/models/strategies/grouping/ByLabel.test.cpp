#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/grouping/ByLabel.hpp"
#include "models/strategies/grouping/Grouping.hpp"
#include "models/strategies/NodeContext.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::grouping;
using namespace ppforest2::stats;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(ByLabelGrouping, FromJsonValid) {
  json const j  = {{"name", "by_label"}};
  auto strategy = ByLabel::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(ByLabelGrouping, FromJsonRoundTrip) {
  json const j  = {{"name", "by_label"}};
  auto strategy = ByLabel::from_json(j);

  auto out = strategy->to_json();

  EXPECT_EQ(out, j);
}

TEST(ByLabelGrouping, FromJsonUnknownParam) {
  json const j = {{"name", "by_label"}, {"extra", 0}};
  EXPECT_THROW(ByLabel::from_json(j), std::runtime_error);
}

TEST(ByLabelGrouping, RegistryLookup) {
  json const j  = {{"name", "by_label"}};
  auto strategy = Grouping::from_json(j);
  ASSERT_NE(strategy, nullptr);

  auto out = strategy->to_json();
  EXPECT_EQ(out, j);
}

TEST(ByLabelGrouping, RegistryUnknownStrategy) {
  json const j = {{"name", "unknown_grouping"}};
  EXPECT_THROW(Grouping::from_json(j), std::runtime_error);
}

TEST(ByLabelGrouping, InitCreatesGroupPartition) {
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2);

  ByLabel const bl;
  auto gp = bl.init(y);

  EXPECT_EQ(gp.groups, (std::set<int>{0, 1, 2}));
  EXPECT_EQ(gp.group_size(0) + gp.group_size(1) + gp.group_size(2), 6);
}

TEST(ByLabelGrouping, SplitsBinaryPartition) {
  // 3 original groups remapped to 2 binary groups: {0,1} -> 0, {2} -> 1
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2);
  GroupPartition const gp(y);

  std::map<int, int> const mapping = {{0, 0}, {1, 0}, {2, 1}};
  GroupPartition const binary_y    = gp.remap(mapping);

  GroupId const group_0 = 0;
  GroupId const group_1 = 1;

  ByLabel const bl;
  auto const result = bl.compute(binary_y, group_0, group_1);

  EXPECT_EQ(result.lower.groups, (std::set<int>{0, 1}));
  EXPECT_EQ(result.upper.groups, (std::set<int>{2}));
}

TEST(ByLabelGrouping, AllRowsAccountedFor) {
  // 4 original groups remapped: {0,1} -> 0, {2,3} -> 1
  FeatureMatrix const x = MAT(Feature, rows(8), 1, 2, 3, 4, 5, 6, 7, 8);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2, 3, 3);
  GroupPartition const gp(y);

  std::map<int, int> const mapping = {{0, 0}, {1, 0}, {2, 1}, {3, 1}};
  GroupPartition const binary_y    = gp.remap(mapping);

  ByLabel const bl;
  auto result     = bl.compute(binary_y, 0, 1);
  auto lower_data = result.lower.data(x).eval();
  auto upper_data = result.upper.data(x).eval();

  EXPECT_EQ_DATA(lower_data, (MAT(Feature, rows(4), 1, 2, 3, 4)));
  EXPECT_EQ_DATA(upper_data, (MAT(Feature, rows(4), 5, 6, 7, 8)));
}

TEST(ByLabelGrouping, AllGroupsAccountedFor) {
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2, 3, 3);
  GroupPartition const gp(y);

  std::map<int, int> const mapping = {{0, 0}, {1, 0}, {2, 1}, {3, 1}};
  GroupPartition const binary_y    = gp.remap(mapping);

  ByLabel const bl;
  auto result = bl.compute(binary_y, 0, 1);

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

TEST(ByLabelGrouping, NoOverlap) {
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2);
  GroupPartition const gp(y);

  std::map<int, int> const mapping = {{0, 0}, {1, 1}, {2, 1}};
  GroupPartition const binary_y    = gp.remap(mapping);

  ByLabel const bl;
  auto const result = bl.compute(binary_y, 0, 1);


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

TEST(ByLabelGrouping, NodeContextInterface) {
  // Test the NodeContext-based split() method
  FeatureMatrix const x = MAT(Feature, rows(6), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2);
  GroupPartition const gp(y);
  RNG rng(0);

  NodeContext ctx(x, gp, 0);

  std::map<int, int> const mapping = {{0, 0}, {1, 0}, {2, 1}};
  ctx.binary_y.emplace(gp.remap(mapping));
  ctx.binary_0 = 0;
  ctx.binary_1 = 1;

  ByLabel const bl;
  auto result = bl.split(ctx, rng);

  EXPECT_EQ(result.lower.groups, (std::set<int>{0, 1}));
  EXPECT_EQ(result.upper.groups, (std::set<int>{2}));
}

TEST(ByLabelGrouping, DisplayName) {
  ByLabel const bl;
  EXPECT_EQ(bl.display_name(), "By label");
}
