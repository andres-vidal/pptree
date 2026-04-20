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
  auto y_part = bl.init(y);

  EXPECT_EQ(y_part.groups, (std::set<int>{0, 1, 2}));
  EXPECT_EQ(y_part.group_size(0) + y_part.group_size(1) + y_part.group_size(2), 6);
}

TEST(ByLabelGrouping, SplitsBinaryPartition) {
  // 3 original groups remapped to 2 binary groups: {0,1} -> 0, {2} -> 1
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2);
  GroupPartition const y_part(y);

  std::map<int, int> const mapping = {{0, 0}, {1, 0}, {2, 1}};
  GroupPartition const y_part_bin  = y_part.remap(mapping);

  ByLabel const bl;
  auto const [lower, upper] = bl.compute(y_part_bin, 0, 1);

  EXPECT_EQ(lower.groups, (std::set<int>{0, 1}));
  EXPECT_EQ(upper.groups, (std::set<int>{2}));
}

TEST(ByLabelGrouping, AllRowsAccountedFor) {
  // 4 original groups remapped: {0,1} -> 0, {2,3} -> 1
  FeatureMatrix x       = MAT(Feature, rows(8), 1, 2, 3, 4, 5, 6, 7, 8);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2, 3, 3);
  GroupPartition const y_part(y);

  std::map<int, int> const mapping = {{0, 0}, {1, 0}, {2, 1}, {3, 1}};
  GroupPartition const y_part_bin  = y_part.remap(mapping);

  ByLabel const bl;
  auto const [lower, upper] = bl.compute(y_part_bin, 0, 1);
  auto lower_data           = lower.data(x).eval();
  auto upper_data           = upper.data(x).eval();

  EXPECT_EQ_DATA(lower_data, (MAT(Feature, rows(4), 1, 2, 3, 4)));
  EXPECT_EQ_DATA(upper_data, (MAT(Feature, rows(4), 5, 6, 7, 8)));
}

TEST(ByLabelGrouping, AllGroupsAccountedFor) {
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2, 3, 3);
  GroupPartition const y_part(y);

  std::map<int, int> const mapping = {{0, 0}, {1, 0}, {2, 1}, {3, 1}};
  GroupPartition const y_part_bin  = y_part.remap(mapping);

  ByLabel const bl;
  auto const [lower, upper] = bl.compute(y_part_bin, 0, 1);

  std::set<int> u;
  std::set_union(
      lower.groups.begin(), lower.groups.end(), upper.groups.begin(), upper.groups.end(), std::inserter(u, u.begin())
  );

  EXPECT_EQ(u, y_part.groups);
}

TEST(ByLabelGrouping, NoOverlap) {
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2);
  GroupPartition const y_part(y);

  std::map<int, int> const mapping = {{0, 0}, {1, 1}, {2, 1}};
  GroupPartition const y_part_bin  = y_part.remap(mapping);

  ByLabel const bl;
  auto const [lower, upper] = bl.compute(y_part_bin, 0, 1);


  std::set<int> i;
  std::set_intersection(
      lower.groups.begin(), lower.groups.end(), upper.groups.begin(), upper.groups.end(), std::inserter(i, i.begin())
  );

  EXPECT_EQ(i, std::set<int>()) << "Groups in lower and upper should not overlap";
}

TEST(ByLabelGrouping, NodeContextInterface) {
  // Test the NodeContext-based split() method
  FeatureMatrix x       = MAT(Feature, rows(6), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2);
  GroupPartition const y_part(y);
  RNG rng(0);

  OutcomeVector ov = y.cast<Outcome>();
  NodeContext ctx(x, y_part, ov, 0);

  std::map<int, int> const mapping = {{0, 0}, {1, 0}, {2, 1}};
  ctx.y_bin.emplace(y_part.remap(mapping));

  ByLabel const bl;
  bl.split(ctx, 0, 1, rng);

  ASSERT_TRUE(ctx.lower_y_part.has_value());
  ASSERT_TRUE(ctx.upper_y_part.has_value());
  EXPECT_EQ(ctx.lower_y_part->groups, (std::set<int>{0, 1})); // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_EQ(ctx.upper_y_part->groups, (std::set<int>{2}));    // NOLINT(bugprone-unchecked-optional-access)
}

TEST(ByLabelGrouping, DisplayName) {
  ByLabel const bl;
  EXPECT_EQ(bl.display_name(), "By label");
}
