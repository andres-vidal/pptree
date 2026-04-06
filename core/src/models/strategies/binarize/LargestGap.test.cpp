#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/binarize/LargestGap.hpp"
#include "models/strategies/binarize/Binarization.hpp"
#include "models/strategies/NodeContext.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::binarize;
using namespace ppforest2::stats;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(LargestGapBinarize, FromJsonValid) {
  json const j  = {{"name", "largest_gap"}};
  auto strategy = LargestGap::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(LargestGapBinarize, FromJsonRoundTrip) {
  json const j  = {{"name", "largest_gap"}};
  auto strategy = LargestGap::from_json(j);

  auto out = strategy->to_json();

  EXPECT_EQ(j, out);
}

TEST(LargestGapBinarize, FromJsonUnknownParam) {
  json const j = {{"name", "largest_gap"}, {"extra", true}};
  EXPECT_THROW(LargestGap::from_json(j), std::runtime_error);
}

TEST(LargestGapBinarize, RegistryLookup) {
  json const j  = {{"name", "largest_gap"}};
  auto strategy = Binarization::from_json(j);
  ASSERT_NE(strategy, nullptr);

  auto out = strategy->to_json();

  EXPECT_EQ(j, out);
}

TEST(LargestGapBinarize, RegistryUnknownStrategy) {
  json const j = {{"name", "unknown_binarize"}};
  EXPECT_THROW(Binarization::from_json(j), std::runtime_error);
}


TEST(LargestGapBinarize, ThreeGroupsSplitByLargestGap) {
  // Three groups with projected means: group 0 -> 1.0, group 1 -> 2.0, group 2 -> 10.0
  // Largest gap is between group 1 (2.0) and group 2 (10.0)
  // So binary group 0 = {0, 1}, binary group 1 = {2}
  FeatureMatrix const projected_x = MAT(Feature, rows(6), 1.0, 1.0, 2.0, 2.0, 10.0, 10.0);
  OutcomeVector const y           = VEC(Outcome, 0, 0, 1, 1, 2, 2);
  GroupPartition const gp(y);

  LargestGap const lg;
  auto result = lg.compute(projected_x, gp);

  EXPECT_EQ(result.binary_y.groups.size(), 2U);

  auto const group_0   = result.group_0;
  auto const group_0_x = result.binary_y.group(projected_x, group_0).eval();
  auto const group_0_y = result.binary_y.group(y, group_0).eval();

  EXPECT_EQ(group_0, 0);
  EXPECT_EQ_DATA(group_0_x, (MAT(Feature, rows(4), 1.0, 1.0, 2.0, 2.0)));
  EXPECT_EQ_DATA(group_0_y, (VEC(Outcome, 0, 0, 1, 1)));

  auto const group_1   = result.group_1;
  auto const group_1_x = result.binary_y.group(projected_x, group_1).eval();
  auto const group_1_y = result.binary_y.group(y, group_1).eval();

  EXPECT_EQ(group_1, 1);
  EXPECT_EQ_DATA(group_1_x, (MAT(Feature, rows(2), 10.0, 10.0)));
  EXPECT_EQ_DATA(group_1_y, (VEC(Outcome, 2, 2)));
}

TEST(LargestGapBinarize, ThreeGroupsPreservesAllObservations) {
  // Three groups: means at 1.0, 5.0, 10.0. Largest gap between 5.0 and 10.0.
  // Binary group 0 = {0, 1}, binary group 1 = {2}
  FeatureMatrix const projected_x = MAT(Feature, rows(6), 1.0, 1.0, 2.0, 2.0, 10.0, 10.0);
  OutcomeVector const y           = VEC(Outcome, 0, 0, 1, 1, 2, 2);
  GroupPartition const gp(y);

  LargestGap const lg;
  auto result     = lg.compute(projected_x, gp);
  auto const data = result.binary_y.data(projected_x).eval();

  EXPECT_EQ_DATA(data, projected_x);
}

TEST(LargestGapBinarize, FourGroupsSplitCorrectly) {
  // Four groups: means at 1, 2, 3, 100 -> largest gap between 3 and 100
  FeatureMatrix const projected_x = MAT(Feature, rows(8), 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 100.0, 100.0);
  OutcomeVector const y           = VEC(Outcome, 0, 0, 1, 1, 2, 2, 3, 3);

  GroupPartition const gp(y);

  LargestGap const lg;
  auto result = lg.compute(projected_x, gp);

  EXPECT_EQ(result.binary_y.groups.size(), 2U);

  auto const group_0   = result.group_0;
  auto const group_0_x = result.binary_y.group(projected_x, group_0).eval();
  auto const group_0_y = result.binary_y.group(y, group_0).eval();

  EXPECT_EQ(group_0, 0);
  EXPECT_EQ_DATA(group_0_x, (MAT(Feature, rows(6), 1.0, 1.0, 2.0, 2.0, 3.0, 3.0)));
  EXPECT_EQ_DATA(group_0_y, (VEC(Outcome, 0, 0, 1, 1, 2, 2)));

  auto const group_1   = result.group_1;
  auto const group_1_x = result.binary_y.group(projected_x, group_1).eval();
  auto const group_1_y = result.binary_y.group(y, group_1).eval();

  EXPECT_EQ(group_1, 1);
  EXPECT_EQ_DATA(group_1_x, (MAT(Feature, rows(2), 100.0, 100.0)));
  EXPECT_EQ_DATA(group_1_y, (VEC(Outcome, 3, 3)));

  auto const data = result.binary_y.data(projected_x).eval();

  EXPECT_EQ_DATA(data, projected_x);
}

TEST(LargestGapBinarize, EqualMeansDoesNotCrash) {
  // All groups have the same projected mean -> all gaps are 0
  FeatureMatrix const projected_x = MAT(Feature, rows(6), 5.0, 5.0, 5.0, 5.0, 5.0, 5.0);
  OutcomeVector const y           = VEC(Outcome, 0, 0, 1, 1, 2, 2);

  GroupPartition const gp(y);

  LargestGap const lg;
  auto result = lg.compute(projected_x, gp);

  // Should still produce a valid binary partition
  ASSERT_EQ(result.binary_y.groups.size(), 2U);
}

TEST(LargestGapBinarize, NodeContextInterface) {
  // Test the NodeContext-based regroup() method
  // 3 groups, feature space where group 2 is far away on dimension 0
  FeatureMatrix const x = MAT(Feature, rows(6), 1, 0, 2, 0, 3, 0, 4, 0, 100, 0, 101, 0);
  OutcomeVector const y = VEC(Outcome, 0, 0, 1, 1, 2, 2);
  GroupPartition const gp(y);
  RNG rng(0);

  NodeContext ctx(x, gp, 0);
  // Set projector to [1, 0] to project onto first dimension
  ctx.projector = VEC(Feature, 1, 0);

  LargestGap const lg;
  lg.regroup(ctx, rng);

  ASSERT_TRUE(ctx.binary_y.has_value());
  EXPECT_EQ(ctx.binary_y->groups.size(), 2U); // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_NE(ctx.binary_0, -1);
  EXPECT_NE(ctx.binary_1, -1);
  EXPECT_NE(ctx.binary_0, ctx.binary_1);
}

TEST(LargestGapBinarize, DisplayName) {
  LargestGap const lg;
  EXPECT_EQ(lg.display_name(), "Largest gap");
}
