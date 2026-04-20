#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/pp/ProjectionPursuit.hpp"
#include "models/strategies/pp/PDA.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::pp;
using namespace ppforest2::stats;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(PPPDAStrategy, FromJsonValid) {
  json const j  = {{"name", "pda"}, {"lambda", 0.3F}};
  auto strategy = PDA::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(PPPDAStrategy, FromJsonRoundTrip) {
  json const j  = {{"name", "pda"}, {"lambda", 0.3F}};
  auto strategy = PDA::from_json(j);

  auto out = strategy->to_json();

  EXPECT_EQ(out, j);
}

TEST(PPPDAStrategy, FromJsonMissingLambda) {
  json const j = {{"name", "pda"}};
  EXPECT_THROW(PDA::from_json(j), std::exception);
}

TEST(PPPDAStrategy, FromJsonUnknownParam) {
  json const j = {{"name", "pda"}, {"lambda", 0.3F}, {"unknown", 1}};
  EXPECT_THROW(PDA::from_json(j), std::runtime_error);
}

TEST(PPPDAStrategy, RegistryLookup) {
  json const j  = {{"name", "pda"}, {"lambda", 0.5F}};
  auto strategy = ProjectionPursuit::from_json(j);
  ASSERT_NE(strategy, nullptr);

  auto out = strategy->to_json();

  EXPECT_EQ(out, j);
}

TEST(PPPDAStrategy, RegistryUnknownStrategy) {
  json const j = {{"name", "unknown_strategy"}};
  EXPECT_THROW(ProjectionPursuit::from_json(j), std::runtime_error);
}


TEST(Projector, LDAOptimumProjectorTwoGroups1) {
  FeatureMatrix x =
      MAT(Feature,
          rows(10),
          1,
          0,
          1,
          1,
          1,
          1,
          0,
          0,
          1,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          4,
          0,
          0,
          1,
          4,
          0,
          0,
          2,
          4,
          0,
          0,
          3,
          4,
          1,
          0,
          1,
          4,
          0,
          1,
          1,
          4,
          0,
          1,
          2);

  GroupIdVector const y = VEC(GroupId, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

  auto result = PDA(0).compute(x, GroupPartition(y));
  auto actual = result.projector;
  auto index  = result.index_value;

  FeatureVector expected = VEC(Feature, -1, 0, 0, 0);

  ASSERT_COLLINEAR(expected, actual);
  ASSERT_FLOAT_EQ(1.0F, index) << "Optimal LDA projector for two groups has index 1";
}

TEST(Projector, LDAOptimumProjectorTwoGroups2) {
  FeatureMatrix x =
      MAT(Feature,
          rows(10),
          0,
          1,
          1,
          1,
          1,
          1,
          0,
          0,
          0,
          1,
          0,
          1,
          1,
          1,
          1,
          1,
          0,
          4,
          0,
          1,
          0,
          4,
          0,
          2,
          0,
          4,
          0,
          3,
          1,
          4,
          0,
          1,
          0,
          4,
          1,
          1,
          0,
          4,
          1,
          2);


  GroupIdVector const y = VEC(GroupId, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

  auto result = PDA(0).compute(x, GroupPartition(y));
  auto actual = result.projector;
  auto index  = result.index_value;

  FeatureVector const expected = VEC(Feature, 0, 1, 0, 0);


  ASSERT_COLLINEAR(expected, actual);
  ASSERT_FLOAT_EQ(1.0F, index) << "Optimal LDA projector for two groups has index 1";
}

TEST(Projector, LDAOptimumProjectorTwoGroups3) {
  FeatureMatrix x =
      MAT(Feature,
          rows(10),
          0,
          1,
          1,
          1,
          1,
          0,
          1,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1,
          0,
          0,
          4,
          1,
          0,
          0,
          4,
          2,
          0,
          0,
          4,
          3,
          1,
          0,
          4,
          1,
          0,
          1,
          4,
          1,
          0,
          1,
          4,
          2);


  GroupIdVector const y = VEC(GroupId, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

  auto result = PDA(0).compute(x, GroupPartition(y));
  auto actual = result.projector;
  auto index  = result.index_value;

  FeatureVector const expected = VEC(Feature, 0, 0, -1, 0);

  ASSERT_COLLINEAR(expected, actual);
  ASSERT_FLOAT_EQ(1.0F, index) << "Optimal LDA projector for two groups has index 1";
}

TEST(Projector, LDAOptimumProjectorTwoGroups4) {
  FeatureMatrix x =
      MAT(Feature,
          rows(10),
          0,
          1,
          1,
          1,
          1,
          0,
          0,
          1,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1,
          0,
          0,
          1,
          4,
          0,
          0,
          2,
          4,
          0,
          0,
          3,
          4,
          1,
          0,
          1,
          4,
          0,
          1,
          1,
          4,
          0,
          1,
          2,
          4);

  GroupIdVector const y = VEC(GroupId, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

  auto result = PDA(0).compute(x, GroupPartition(y));
  auto actual = result.projector;
  auto index  = result.index_value;

  FeatureVector const expected =
      VEC(Feature, 2.0965219514666735e-15, 4.4408920985006262e-16, -2.4980018054066002e-16, 1);

  ASSERT_COLLINEAR(expected, actual);
  ASSERT_FLOAT_EQ(1.0F, index) << "Optimal LDA projector for two groups has index 1";
}

TEST(Projector, LDAOptimumProjectorThreeGroups1) {
  FeatureMatrix x =
      MAT(Feature,
          rows(30),
          1,
          0,
          0,
          1,
          1,
          1,
          0,
          1,
          0,
          0,
          1,
          0,
          0,
          0,
          1,
          1,
          0,
          1,
          1,
          1,
          1,
          0,
          0,
          1,
          1,
          1,
          0,
          1,
          1,
          0,
          1,
          0,
          0,
          1,
          1,
          1,
          0,
          1,
          1,
          2,
          1,
          0,
          0,
          2,
          0,
          1,
          0,
          2,
          1,
          0,
          2,
          8,
          0,
          0,
          1,
          2,
          8,
          0,
          0,
          2,
          2,
          8,
          1,
          0,
          2,
          2,
          8,
          1,
          0,
          1,
          2,
          8,
          0,
          1,
          1,
          2,
          8,
          0,
          1,
          2,
          2,
          8,
          2,
          1,
          1,
          2,
          8,
          1,
          1,
          1,
          2,
          8,
          1,
          1,
          2,
          2,
          8,
          2,
          1,
          2,
          2,
          8,
          1,
          2,
          1,
          2,
          8,
          2,
          1,
          1,
          9,
          8,
          0,
          0,
          1,
          9,
          8,
          0,
          0,
          2,
          9,
          8,
          1,
          0,
          2,
          9,
          8,
          1,
          0,
          1,
          9,
          8,
          0,
          1,
          1,
          9,
          8,
          0,
          1,
          2,
          9,
          8,
          2,
          1,
          1,
          9,
          8,
          1,
          1,
          1);

  GroupIdVector const y =
      VEC(GroupId, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);

  auto result = PDA(0).compute(x, GroupPartition(y));
  auto actual = result.projector;
  auto index  = result.index_value;

  FeatureVector const expected = VEC(Feature, 0.0351398066F, -0.0574720800F, 0, 0, 0);

  ASSERT_COLLINEAR(expected, actual);
  ASSERT_GT(index, 0.99F) << "Optimal LDA projector for three groups has index near 1";
}

TEST(Projector, PDAOptimumProjectorLambdaOneHalfTwoGroups) {
  FeatureMatrix x = MAT(Feature, rows(4), 1, 0, 1, 1, 1, 4, 2, 1, 0, 0, 0, 4, 3, 0, 1, 1, 1, 1, 4, 0, 1, 2, 2, 1);

  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1);

  auto result = PDA(0.5).compute(x, GroupPartition(y));
  auto actual = result.projector;
  auto index  = result.index_value;

  FeatureVector const expected = VEC(Feature, 0, 0, 0, 0, 0, 1);

  ASSERT_COLLINEAR(expected, actual);
  ASSERT_GT(index, 0.0F) << "PDA optimal projector has positive index";
}

TEST(Projector, PDAOptimumProjectorZeroColumn) {
  FeatureMatrix x =
      MAT(Feature, rows(4), 1, 0, 1, 1, 1, 4, 0, 2, 1, 0, 0, 0, 4, 0, 3, 0, 1, 1, 1, 1, 0, 4, 0, 1, 2, 2, 1, 0);

  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1);

  auto result = PDA(0.1).compute(x, GroupPartition(y));
  auto actual = result.projector;
  auto index  = result.index_value;

  ASSERT_TRUE(actual.hasNaN()) << "Zero column with tiny sample produces degenerate (NaN) projector";
  ASSERT_TRUE(std::isnan(index)) << "Degenerate projector has NaN index";
}

TEST(Projector, PDALambdaOneBoundary) {
  FeatureMatrix x =
      MAT(Feature,
          rows(10),
          1,
          0,
          1,
          1,
          1,
          1,
          0,
          0,
          1,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          4,
          0,
          0,
          1,
          4,
          0,
          0,
          2,
          4,
          0,
          0,
          3,
          4,
          1,
          0,
          1,
          4,
          0,
          1,
          1,
          4,
          0,
          1,
          2);

  GroupIdVector const y = VEC(GroupId, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

  auto result = PDA(1.0).compute(x, GroupPartition(y));
  auto actual = result.projector;
  auto index  = result.index_value;

  // Lambda=1 means full penalization (diagonal covariance)
  // Projector should still point in the discriminating direction
  FeatureVector const expected = VEC(Feature, 1, 0, 0, 0);
  ASSERT_COLLINEAR(expected, actual);
  ASSERT_GT(index, 0.0F) << "PDA lambda=1 should still find a valid projector";
}
