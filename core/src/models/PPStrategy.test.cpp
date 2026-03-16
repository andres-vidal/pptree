#include <gtest/gtest.h>

#include "models/PPStrategy.hpp"
#include "models/PPGLDAStrategy.hpp"
#include "utils/Types.hpp"

#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::pp;
using namespace ppforest2::stats;
using namespace ppforest2::types;

TEST(Projector, LDAOptimumProjectorTwoGroups1) {
  FeatureMatrix x = MAT(Feature, rows(10),
      1, 0, 1, 1,
      1, 1, 0, 0,
      1, 0, 0, 1,
      1, 1, 1, 1,
      4, 0, 0, 1,
      4, 0, 0, 2,
      4, 0, 0, 3,
      4, 1, 0, 1,
      4, 0, 1, 1,
      4, 0, 1, 2);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1);

  auto [actual, index] = PPGLDAStrategy(0).optimize(x, GroupPartition(y));

  FeatureVector expected = VEC(Feature,
      -1, 0, 0, 0);

  ASSERT_COLLINEAR(expected, actual);
  ASSERT_FLOAT_EQ(1.0f, index) << "Optimal LDA projector for two groups has index 1";
}

TEST(Projector, LDAOptimumProjectorTwoGroups2) {
  FeatureMatrix x = MAT(Feature, rows(10),
      0, 1, 1, 1,
      1, 1, 0, 0,
      0, 1, 0, 1,
      1, 1, 1, 1,
      0, 4, 0, 1,
      0, 4, 0, 2,
      0, 4, 0, 3,
      1, 4, 0, 1,
      0, 4, 1, 1,
      0, 4, 1, 2);


  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1);

  auto [actual, index] = PPGLDAStrategy(0).optimize(x, GroupPartition(y));

  FeatureVector expected = VEC(Feature,
      0, 1, 0, 0);


  ASSERT_COLLINEAR(expected, actual);
  ASSERT_FLOAT_EQ(1.0f, index) << "Optimal LDA projector for two groups has index 1";
}

TEST(Projector, LDAOptimumProjectorTwoGroups3) {
  FeatureMatrix x = MAT(Feature, rows(10),
      0, 1, 1, 1,
      1, 0, 1, 0,
      0, 0, 1, 1,
      1, 1, 1, 1,
      0, 0, 4, 1,
      0, 0, 4, 2,
      0, 0, 4, 3,
      1, 0, 4, 1,
      0, 1, 4, 1,
      0, 1, 4, 2);


  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1);

  auto [actual, index] = PPGLDAStrategy(0).optimize(x, GroupPartition(y));

  FeatureVector expected = VEC(Feature,
      0, 0, -1, 0);

  ASSERT_COLLINEAR(expected, actual);
  ASSERT_FLOAT_EQ(1.0f, index) << "Optimal LDA projector for two groups has index 1";
}

TEST(Projector, LDAOptimumProjectorTwoGroups4) {
  FeatureMatrix x = MAT(Feature, rows(10),
      0, 1, 1, 1,
      1, 0, 0, 1,
      0, 0, 1, 1,
      1, 1, 1, 1,
      0, 0, 1, 4,
      0, 0, 2, 4,
      0, 0, 3, 4,
      1, 0, 1, 4,
      0, 1, 1, 4,
      0, 1, 2, 4);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1);

  auto [actual, index] = PPGLDAStrategy(0).optimize(x, GroupPartition(y));

  FeatureVector expected = VEC(Feature,
      2.0965219514666735e-15,
      4.4408920985006262e-16,
      -2.4980018054066002e-16,
      1);

  ASSERT_COLLINEAR(expected, actual);
  ASSERT_FLOAT_EQ(1.0f, index) << "Optimal LDA projector for two groups has index 1";
}

TEST(Projector, LDAOptimumProjectorThreeGroups1) {
  FeatureMatrix x = MAT(Feature, rows(30),
      1, 0, 0, 1, 1,
      1, 0, 1, 0, 0,
      1, 0, 0, 0, 1,
      1, 0, 1, 1, 1,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 0,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 2,
      1, 0, 0, 2, 0,
      1, 0, 2, 1, 0,
      2, 8, 0, 0, 1,
      2, 8, 0, 0, 2,
      2, 8, 1, 0, 2,
      2, 8, 1, 0, 1,
      2, 8, 0, 1, 1,
      2, 8, 0, 1, 2,
      2, 8, 2, 1, 1,
      2, 8, 1, 1, 1,
      2, 8, 1, 1, 2,
      2, 8, 2, 1, 2,
      2, 8, 1, 2, 1,
      2, 8, 2, 1, 1,
      9, 8, 0, 0, 1,
      9, 8, 0, 0, 2,
      9, 8, 1, 0, 2,
      9, 8, 1, 0, 1,
      9, 8, 0, 1, 1,
      9, 8, 0, 1, 2,
      9, 8, 2, 1, 1,
      9, 8, 1, 1, 1);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2);

  auto [actual, index] = PPGLDAStrategy(0).optimize(x, GroupPartition(y));

  FeatureVector expected = VEC(Feature,
      0.0351398066f,
      -0.0574720800f,
      0,
      0,
      0);

  ASSERT_COLLINEAR(expected, actual);
  ASSERT_GT(index, 0.99f) << "Optimal LDA projector for three groups has index near 1";
}

TEST(Projector, LDAIndexZeroReturn) {
  FeatureMatrix x = MAT(Feature, rows(4),
      1, 0, 1, 1, 0, 1, 0, 1, 2, 3, 4, 5,
      1, 1, 0, 0, 0, 1, 0, 1, 2, 3, 4, 5,
      1, 1, 0, 1, 1, 0, 1, 0, 2, 3, 4, 5,
      1, 0, 1, 1, 1, 0, 1, 0, 2, 3, 4, 5);

  ResponseVector y = VEC(Response,
      0,
      0,
      1,
      1);

  FeatureVector projector = VEC(Feature,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

  float actual = PPGLDAStrategy(0).index(x, GroupPartition(y), projector);

  ASSERT_EQ(0.0, actual);
}

TEST(Projector, LDAIndexOptimal1) {
  FeatureMatrix x = MAT(Feature, rows(30),
      1, 0, 0, 1, 1,
      1, 0, 1, 0, 0,
      1, 0, 0, 0, 1,
      1, 0, 1, 1, 1,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 0,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 2,
      1, 0, 0, 2, 0,
      1, 0, 2, 1, 0,
      2, 8, 0, 0, 1,
      2, 8, 0, 0, 2,
      2, 8, 1, 0, 2,
      2, 8, 1, 0, 1,
      2, 8, 0, 1, 1,
      2, 8, 0, 1, 2,
      2, 8, 2, 1, 1,
      2, 8, 1, 1, 1,
      2, 8, 1, 1, 2,
      2, 8, 2, 1, 2,
      2, 8, 1, 2, 1,
      2, 8, 2, 1, 1,
      9, 8, 0, 0, 1,
      9, 8, 0, 0, 2,
      9, 8, 1, 0, 2,
      9, 8, 1, 0, 1,
      9, 8, 0, 1, 1,
      9, 8, 0, 1, 2,
      9, 8, 2, 1, 1,
      9, 8, 1, 1, 1);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2);

  FeatureVector projector = VEC(Feature,
      -0.12823, -0.99174, 0.0, 0.0, 0.0);


  float actual = PPGLDAStrategy(0).index(x, GroupPartition(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, LDAIndexOptimal2) {
  FeatureMatrix x = MAT(Feature, rows(30),
      1, 0, 0, 1, 1,
      1, 0, 1, 0, 0,
      1, 0, 0, 0, 1,
      1, 0, 1, 1, 1,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 0,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 2,
      1, 0, 0, 2, 0,
      1, 0, 2, 1, 0,
      2, 8, 0, 0, 1,
      2, 8, 0, 0, 2,
      2, 8, 1, 0, 2,
      2, 8, 1, 0, 1,
      2, 8, 0, 1, 1,
      2, 8, 0, 1, 2,
      2, 8, 2, 1, 1,
      2, 8, 1, 1, 1,
      2, 8, 1, 1, 2,
      2, 8, 2, 1, 2,
      2, 8, 1, 2, 1,
      2, 8, 2, 1, 1,
      9, 8, 0, 0, 1,
      9, 8, 0, 0, 2,
      9, 8, 1, 0, 2,
      9, 8, 1, 0, 1,
      9, 8, 0, 1, 1,
      9, 8, 0, 1, 2,
      9, 8, 2, 1, 1,
      9, 8, 1, 1, 1);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2);

  FeatureVector projector = VEC(Feature,
      0.78481, 0.61974, 0.0, 0.0, 0.0);

  float actual = PPGLDAStrategy(0).index(x, GroupPartition(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, LDAIndexOptimal3) {
  FeatureMatrix x = MAT(Feature, rows(30),
      1, 0, 0, 1, 1,
      1, 0, 1, 0, 0,
      1, 0, 0, 0, 1,
      1, 0, 1, 1, 1,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 0,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 2,
      1, 0, 0, 2, 0,
      1, 0, 2, 1, 0,
      2, 8, 0, 0, 1,
      2, 8, 0, 0, 2,
      2, 8, 1, 0, 2,
      2, 8, 1, 0, 1,
      2, 8, 0, 1, 1,
      2, 8, 0, 1, 2,
      2, 8, 2, 1, 1,
      2, 8, 1, 1, 1,
      2, 8, 1, 1, 2,
      2, 8, 2, 1, 2,
      2, 8, 1, 2, 1,
      2, 8, 2, 1, 1,
      9, 8, 0, 0, 1,
      9, 8, 0, 0, 2,
      9, 8, 1, 0, 2,
      9, 8, 1, 0, 1,
      9, 8, 0, 1, 1,
      9, 8, 0, 1, 2,
      9, 8, 2, 1, 1,
      9, 8, 1, 1, 1);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2);

  FeatureVector projector = VEC(Feature,
      -0.66808,  0.74409,  0.0,  0.0,  0.0);

  float actual = PPGLDAStrategy(0).index(x, GroupPartition(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, LDAIndexSuboptimal1) {
  FeatureMatrix x = MAT(Feature, rows(30),
      1, 0, 0, 1, 1,
      1, 0, 1, 0, 0,
      1, 0, 0, 0, 1,
      1, 0, 1, 1, 1,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 0,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 2,
      1, 0, 0, 2, 0,
      1, 0, 2, 1, 0,
      2, 8, 0, 0, 1,
      2, 8, 0, 0, 2,
      2, 8, 1, 0, 2,
      2, 8, 1, 0, 1,
      2, 8, 0, 1, 1,
      2, 8, 0, 1, 2,
      2, 8, 2, 1, 1,
      2, 8, 1, 1, 1,
      2, 8, 1, 1, 2,
      2, 8, 2, 1, 2,
      2, 8, 1, 2, 1,
      2, 8, 2, 1, 1,
      9, 8, 0, 0, 1,
      9, 8, 0, 0, 2,
      9, 8, 1, 0, 2,
      9, 8, 1, 0, 1,
      9, 8, 0, 1, 1,
      9, 8, 0, 1, 2,
      9, 8, 2, 1, 1,
      9, 8, 1, 1, 1);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2);

  FeatureVector projector = VEC(Feature,
      0, 0, 1, 1, 1);

  float actual = PPGLDAStrategy(0).index(x, GroupPartition(y), projector);

  ASSERT_NEAR(0.134985, actual, 0.00001);
}

TEST(Projector, LDAIndexSuboptimal2) {
  FeatureMatrix x = MAT(Feature, rows(30),
      1, 0, 0, 1, 1,
      1, 0, 1, 0, 0,
      1, 0, 0, 0, 1,
      1, 0, 1, 1, 1,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 0,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 2,
      1, 0, 0, 2, 0,
      1, 0, 2, 1, 0,
      2, 8, 0, 0, 1,
      2, 8, 0, 0, 2,
      2, 8, 1, 0, 2,
      2, 8, 1, 0, 1,
      2, 8, 0, 1, 1,
      2, 8, 0, 1, 2,
      2, 8, 2, 1, 1,
      2, 8, 1, 1, 1,
      2, 8, 1, 1, 2,
      2, 8, 2, 1, 2,
      2, 8, 1, 2, 1,
      2, 8, 2, 1, 1,
      9, 8, 0, 0, 1,
      9, 8, 0, 0, 2,
      9, 8, 1, 0, 2,
      9, 8, 1, 0, 1,
      9, 8, 0, 1, 1,
      9, 8, 0, 1, 2,
      9, 8, 2, 1, 1,
      9, 8, 1, 1, 1);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2);

  FeatureVector projector = VEC(Feature,
      -0.02965,  0.08452, -0.24243, -0.40089, -0.87892);

  float actual = PPGLDAStrategy(0).index(x, GroupPartition(y), projector);

  ASSERT_NEAR(0.0, actual, 0.000001);
}

TEST(Projector, PDAOptimumProjectorLambdaOneHalfTwoGroups) {
  FeatureMatrix x = MAT(Feature, rows(4),
      1, 0, 1, 1, 1, 4,
      2, 1, 0, 0, 0, 4,
      3, 0, 1, 1, 1, 1,
      4, 0, 1, 2, 2, 1);

  ResponseVector y = VEC(Response,
      0,
      0,
      1,
      1);

  auto [actual, index] = PPGLDAStrategy(0.5).optimize(x, GroupPartition(y));

  FeatureVector expected = VEC(Feature,
      0, 0, 0, 0, 0, 1);

  ASSERT_COLLINEAR(expected, actual);
  ASSERT_GT(index, 0.0f) << "PDA optimal projector has positive index";
}

TEST(Projector, GLDAOptimumProjectorZeroColumn) {
  FeatureMatrix x = MAT(Feature, rows(4),
      1, 0, 1, 1, 1, 4, 0,
      2, 1, 0, 0, 0, 4, 0,
      3, 0, 1, 1, 1, 1, 0,
      4, 0, 1, 2, 2, 1, 0);

  ResponseVector y = VEC(Response,
      0,
      0,
      1,
      1);

  auto [actual, index] = PPGLDAStrategy(0.1).optimize(x, GroupPartition(y));

  ASSERT_TRUE(actual.hasNaN()) << "Zero column with tiny sample produces degenerate (NaN) projector";
  ASSERT_TRUE(std::isnan(index)) << "Degenerate projector has NaN index";
}

TEST(Projector, PDAIndexLambdaOneHalfZeroReturn) {
  FeatureMatrix x = MAT(Feature, rows(4),
      1, 0, 1, 1, 0, 1, 0, 1, 2, 3, 4, 5,
      1, 1, 0, 0, 0, 1, 0, 1, 2, 3, 4, 5,
      1, 1, 0, 1, 1, 0, 1, 0, 2, 3, 4, 5,
      1, 0, 1, 1, 1, 0, 1, 0, 2, 3, 4, 5);

  ResponseVector y = VEC(Response,
      0,
      0,
      1,
      1);

  FeatureVector projector = VEC(Feature,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);


  float actual = PPGLDAStrategy(0.5).index(x, GroupPartition(y), projector);

  ASSERT_EQ(0.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfOptimal1) {
  FeatureMatrix x = MAT(Feature, rows(30),
      1, 0, 0, 1, 1,
      1, 0, 1, 0, 0,
      1, 0, 0, 0, 1,
      1, 0, 1, 1, 1,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 0,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 2,
      1, 0, 0, 2, 0,
      1, 0, 2, 1, 0,
      2, 8, 0, 0, 1,
      2, 8, 0, 0, 2,
      2, 8, 1, 0, 2,
      2, 8, 1, 0, 1,
      2, 8, 0, 1, 1,
      2, 8, 0, 1, 2,
      2, 8, 2, 1, 1,
      2, 8, 1, 1, 1,
      2, 8, 1, 1, 2,
      2, 8, 2, 1, 2,
      2, 8, 1, 2, 1,
      2, 8, 2, 1, 1,
      9, 8, 0, 0, 1,
      9, 8, 0, 0, 2,
      9, 8, 1, 0, 2,
      9, 8, 1, 0, 1,
      9, 8, 0, 1, 1,
      9, 8, 0, 1, 2,
      9, 8, 2, 1, 1,
      9, 8, 1, 1, 1);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2);

  FeatureVector projector = VEC(Feature,
      -0.12823, -0.99174, 0.0, 0.0, 0.0);

  float actual = PPGLDAStrategy(0.5).index(x, GroupPartition(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfOptimal2) {
  FeatureMatrix x = MAT(Feature, rows(30),
      1, 0, 0, 1, 1,
      1, 0, 1, 0, 0,
      1, 0, 0, 0, 1,
      1, 0, 1, 1, 1,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 0,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 2,
      1, 0, 0, 2, 0,
      1, 0, 2, 1, 0,
      2, 8, 0, 0, 1,
      2, 8, 0, 0, 2,
      2, 8, 1, 0, 2,
      2, 8, 1, 0, 1,
      2, 8, 0, 1, 1,
      2, 8, 0, 1, 2,
      2, 8, 2, 1, 1,
      2, 8, 1, 1, 1,
      2, 8, 1, 1, 2,
      2, 8, 2, 1, 2,
      2, 8, 1, 2, 1,
      2, 8, 2, 1, 1,
      9, 8, 0, 0, 1,
      9, 8, 0, 0, 2,
      9, 8, 1, 0, 2,
      9, 8, 1, 0, 1,
      9, 8, 0, 1, 1,
      9, 8, 0, 1, 2,
      9, 8, 2, 1, 1,
      9, 8, 1, 1, 1);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2);

  FeatureVector projector = VEC(Feature,
      0.78481, 0.61974, 0.0, 0.0, 0.0);

  float actual = PPGLDAStrategy(0.5).index(x, GroupPartition(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfOptimal3) {
  FeatureMatrix x = MAT(Feature, rows(30),
      1, 0, 0, 1, 1,
      1, 0, 1, 0, 0,
      1, 0, 0, 0, 1,
      1, 0, 1, 1, 1,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 0,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 2,
      1, 0, 0, 2, 0,
      1, 0, 2, 1, 0,
      2, 8, 0, 0, 1,
      2, 8, 0, 0, 2,
      2, 8, 1, 0, 2,
      2, 8, 1, 0, 1,
      2, 8, 0, 1, 1,
      2, 8, 0, 1, 2,
      2, 8, 2, 1, 1,
      2, 8, 1, 1, 1,
      2, 8, 1, 1, 2,
      2, 8, 2, 1, 2,
      2, 8, 1, 2, 1,
      2, 8, 2, 1, 1,
      9, 8, 0, 0, 1,
      9, 8, 0, 0, 2,
      9, 8, 1, 0, 2,
      9, 8, 1, 0, 1,
      9, 8, 0, 1, 1,
      9, 8, 0, 1, 2,
      9, 8, 2, 1, 1,
      9, 8, 1, 1, 1);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2);

  FeatureVector projector = VEC(Feature,
      -0.66808,  0.74409,  0.0,  0.0,  0.0);

  float actual = PPGLDAStrategy(0.5).index(x, GroupPartition(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfSubptimal1) {
  FeatureMatrix x = MAT(Feature, rows(30),
      1, 0, 0, 1, 1,
      1, 0, 1, 0, 0,
      1, 0, 0, 0, 1,
      1, 0, 1, 1, 1,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 0,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 2,
      1, 0, 0, 2, 0,
      1, 0, 2, 1, 0,
      2, 8, 0, 0, 1,
      2, 8, 0, 0, 2,
      2, 8, 1, 0, 2,
      2, 8, 1, 0, 1,
      2, 8, 0, 1, 1,
      2, 8, 0, 1, 2,
      2, 8, 2, 1, 1,
      2, 8, 1, 1, 1,
      2, 8, 1, 1, 2,
      2, 8, 2, 1, 2,
      2, 8, 1, 2, 1,
      2, 8, 2, 1, 1,
      9, 8, 0, 0, 1,
      9, 8, 0, 0, 2,
      9, 8, 1, 0, 2,
      9, 8, 1, 0, 1,
      9, 8, 0, 1, 1,
      9, 8, 0, 1, 2,
      9, 8, 2, 1, 1,
      9, 8, 1, 1, 1);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2);

  FeatureVector projector = VEC(Feature,
      0, 0, 1, 1, 1);

  float actual = PPGLDAStrategy(0.5).index(x, GroupPartition(y), projector);

  ASSERT_NEAR(0.12597, actual, 0.00001);
}

TEST(Projector, PDALambdaOneBoundary) {
  FeatureMatrix x = MAT(Feature, rows(10),
      1, 0, 1, 1,
      1, 1, 0, 0,
      1, 0, 0, 1,
      1, 1, 1, 1,
      4, 0, 0, 1,
      4, 0, 0, 2,
      4, 0, 0, 3,
      4, 1, 0, 1,
      4, 0, 1, 1,
      4, 0, 1, 2);

  ResponseVector y = VEC(Response,
      0, 0, 0, 0,
      1, 1, 1, 1, 1, 1);

  auto [actual, index] = PPGLDAStrategy(1.0).optimize(x, GroupPartition(y));

  // Lambda=1 means full penalization (diagonal covariance)
  // Projector should still point in the discriminating direction
  FeatureVector expected = VEC(Feature, 1, 0, 0, 0);
  ASSERT_COLLINEAR(expected, actual);
  ASSERT_GT(index, 0.0f) << "PDA lambda=1 should still find a valid projector";
}

TEST(Projector, PDAIndexLambdaOneHalfSubptimal2) {
  FeatureMatrix x = MAT(Feature, rows(30),
      1, 0, 0, 1, 1,
      1, 0, 1, 0, 0,
      1, 0, 0, 0, 1,
      1, 0, 1, 1, 1,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 0,
      1, 0, 0, 1, 1,
      1, 0, 1, 1, 2,
      1, 0, 0, 2, 0,
      1, 0, 2, 1, 0,
      2, 8, 0, 0, 1,
      2, 8, 0, 0, 2,
      2, 8, 1, 0, 2,
      2, 8, 1, 0, 1,
      2, 8, 0, 1, 1,
      2, 8, 0, 1, 2,
      2, 8, 2, 1, 1,
      2, 8, 1, 1, 1,
      2, 8, 1, 1, 2,
      2, 8, 2, 1, 2,
      2, 8, 1, 2, 1,
      2, 8, 2, 1, 1,
      9, 8, 0, 0, 1,
      9, 8, 0, 0, 2,
      9, 8, 1, 0, 2,
      9, 8, 1, 0, 1,
      9, 8, 0, 1, 1,
      9, 8, 0, 1, 2,
      9, 8, 2, 1, 1,
      9, 8, 1, 1, 1);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2);

  FeatureVector projector = VEC(Feature,
      -0.02965,  0.08452, -0.24243, -0.40089, -0.87892);

  float actual = PPGLDAStrategy(0.5).index(x, GroupPartition(y), projector);

  ASSERT_NEAR(0.0, actual, 0.000001);
}
