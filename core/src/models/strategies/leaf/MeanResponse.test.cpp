#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/leaf/MeanResponse.hpp"
#include "models/strategies/leaf/LeafStrategy.hpp"
#include "models/strategies/NodeContext.hpp"
#include "models/TreeLeaf.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::leaf;
using namespace ppforest2::stats;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(MeanResponseLeaf, FromJsonValid) {
  json const j  = {{"name", "mean_response"}};
  auto strategy = MeanResponse::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(MeanResponseLeaf, FromJsonRoundTrip) {
  json const j  = {{"name", "mean_response"}};
  auto strategy = MeanResponse::from_json(j);

  auto out = strategy->to_json();

  EXPECT_EQ(out, j);
}

TEST(MeanResponseLeaf, FromJsonUnknownParam) {
  json const j = {{"name", "mean_response"}, {"extra", 0}};
  EXPECT_THROW(MeanResponse::from_json(j), std::runtime_error);
}

TEST(MeanResponseLeaf, RegistryLookup) {
  json const j  = {{"name", "mean_response"}};
  auto strategy = LeafStrategy::from_json(j);
  ASSERT_NE(strategy, nullptr);

  auto out = strategy->to_json();
  EXPECT_EQ(out, j);
}

TEST(MeanResponseLeaf, DisplayName) {
  MeanResponse const mr;
  EXPECT_EQ(mr.display_name(), "Mean response");
}

TEST(MeanResponseLeaf, MeanOfAllObservations) {
  FeatureMatrix x       = MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1);
  GroupPartition const gp(y);
  OutcomeVector ov = VEC(Feature, 1.0, 3.0, 5.0, 7.0);
  RNG rng(0);

  NodeContext ctx(x, gp, ov, 0);

  MeanResponse const mr;
  auto leaf = mr.create_leaf(ctx, rng);

  ASSERT_NE(leaf, nullptr);
  EXPECT_TRUE(leaf->is_leaf());
  EXPECT_FLOAT_EQ(static_cast<float>(leaf->response()), 4.0f);
}

TEST(MeanResponseLeaf, MeanOfSingleGroup) {
  FeatureMatrix x       = MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6);
  GroupIdVector const y = VEC(GroupId, 0, 0, 0);
  GroupPartition const gp(y);
  OutcomeVector ov = VEC(Feature, 2.0, 4.0, 6.0);
  RNG rng(0);

  NodeContext ctx(x, gp, ov, 0);

  MeanResponse const mr;
  auto leaf = mr.create_leaf(ctx, rng);

  EXPECT_FLOAT_EQ(static_cast<float>(leaf->response()), 4.0f);
}

TEST(MeanResponseLeaf, MeanOfSubset) {
  FeatureMatrix x       = MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1);
  GroupPartition const gp(y);
  GroupPartition const sub = gp.subset({0});
  OutcomeVector ov         = VEC(Feature, 10.0, 20.0, 30.0, 40.0);
  RNG rng(0);

  NodeContext ctx(x, sub, ov, 0);
  ctx.y_vec = &ov;

  MeanResponse const mr;
  auto leaf = mr.create_leaf(ctx, rng);

  // Only group 0 observations (indices 0,1) → mean of 10.0 and 20.0 = 15.0
  EXPECT_FLOAT_EQ(static_cast<float>(leaf->response()), 15.0f);
}
