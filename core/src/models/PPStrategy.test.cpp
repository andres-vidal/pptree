#include <gtest/gtest.h>

#include "models/PPStrategy.hpp"
#include "models/PPGLDAStrategy.hpp"
#include "utils/Types.hpp"

#include "utils/Macros.hpp"

using namespace pptree;
using namespace pptree::pp;
using namespace pptree::stats;
using namespace pptree::types;

TEST(Projector, LDAOptimumProjectorTwoGroups1) {
  FeatureMatrix x = DATA(Feature, 10,
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

  ResponseVector y = DATA(Response, 10,
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

  FeatureVector actual = PPGLDAStrategy(0).optimize(x, GroupPartition(y));

  FeatureVector expected = DATA(Feature, 4,
      -1, 0, 0, 0);

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAOptimumProjectorTwoGroups2) {
  FeatureMatrix x = DATA(Feature, 10,
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


  ResponseVector y = DATA(Response, 10,
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

  FeatureVector actual = PPGLDAStrategy(0).optimize(x, GroupPartition(y));

  FeatureVector expected = DATA(Feature, 4,
      0, 1, 0, 0);


  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAOptimumProjectorTwoGroups3) {
  FeatureMatrix x = DATA(Feature, 10,
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


  ResponseVector y = DATA(Response, 10,
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

  FeatureVector actual = PPGLDAStrategy(0).optimize(x, GroupPartition(y));

  FeatureVector expected = DATA(Feature, 4,
      0, 0, -1, 0);

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAOptimumProjectorTwoGroups4) {
  FeatureMatrix x = DATA(Feature, 10,
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

  ResponseVector y = DATA(Response, 10,
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

  FeatureVector actual = PPGLDAStrategy(0).optimize(x, GroupPartition(y));

  FeatureVector expected = DATA(Feature, 4,
      2.0965219514666735e-15,
      4.4408920985006262e-16,
      -2.4980018054066002e-16,
      1);

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAOptimumProjectorThreeGroups1) {
  FeatureMatrix x = DATA(Feature, 30,
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

  ResponseVector y = DATA(Response, 30,
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

  FeatureVector actual = PPGLDAStrategy(0).optimize(x, GroupPartition(y));

  FeatureVector expected = DATA(Feature, 5,
      1,
      0,
      0,
      0,
      0);

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAIndexZeroReturn) {
  FeatureMatrix x = DATA(Feature, 4,
      1, 0, 1, 1, 0, 1, 0, 1, 2, 3, 4, 5,
      1, 1, 0, 0, 0, 1, 0, 1, 2, 3, 4, 5,
      1, 1, 0, 1, 1, 0, 1, 0, 2, 3, 4, 5,
      1, 0, 1, 1, 1, 0, 1, 0, 2, 3, 4, 5);

  ResponseVector y = DATA(Response, 4,
      0,
      0,
      1,
      1);

  FeatureVector projector = DATA(Feature, 12,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

  float actual = PPGLDAStrategy(0).index(x, GroupPartition(y), projector);

  ASSERT_EQ(0.0, actual);
}

TEST(Projector, LDAIndexOptimal1) {
  FeatureMatrix x = DATA(Feature, 30,
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

  ResponseVector y = DATA(Response, 30,
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

  FeatureVector projector = DATA(Feature, 5,
      -0.12823, -0.99174, 0.0, 0.0, 0.0);


  float actual = PPGLDAStrategy(0).index(x, GroupPartition(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, LDAIndexOptimal2) {
  FeatureMatrix x = DATA(Feature, 30,
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

  ResponseVector y = DATA(Response, 30,
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

  FeatureVector projector = DATA(Feature, 5,
      0.78481, 0.61974, 0.0, 0.0, 0.0);

  float actual = PPGLDAStrategy(0).index(x, GroupPartition(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, LDAIndexOptimal3) {
  FeatureMatrix x = DATA(Feature, 30,
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

  ResponseVector y = DATA(Response, 30,
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

  FeatureVector projector = DATA(Feature, 5,
      -0.66808,  0.74409,  0.0,  0.0,  0.0);

  float actual = PPGLDAStrategy(0).index(x, GroupPartition(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, LDAIndexSuboptimal1) {
  FeatureMatrix x = DATA(Feature, 30,
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

  ResponseVector y = DATA(Response, 30,
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

  FeatureVector projector = DATA(Feature, 5,
      0, 0, 1, 1, 1);

  float actual = PPGLDAStrategy(0).index(x, GroupPartition(y), projector);

  ASSERT_NEAR(0.134985, actual, 0.00001);
}

TEST(Projector, LDAIndexSuboptimal2) {
  FeatureMatrix x = DATA(Feature, 30,
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

  ResponseVector y = DATA(Response, 30,
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

  FeatureVector projector = DATA(Feature, 5,
      -0.02965,  0.08452, -0.24243, -0.40089, -0.87892);

  float actual = PPGLDAStrategy(0).index(x, GroupPartition(y), projector);

  ASSERT_NEAR(0.0, actual, 0.000001);
}

TEST(Projector, PDAOptimumProjectorLambdaOneHalfTwoGroups) {
  FeatureMatrix x = DATA(Feature, 4,
      1, 0, 1, 1, 1, 4,
      2, 1, 0, 0, 0, 4,
      3, 0, 1, 1, 1, 1,
      4, 0, 1, 2, 2, 1);

  ResponseVector y = DATA(Response, 4,
      0,
      0,
      1,
      1);

  FeatureVector actual = PPGLDAStrategy(0.5).optimize(x, GroupPartition(y));

  FeatureVector expected = DATA(Feature, 6,
      0, 0, 0, 0, 0, 1);

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, GLDAOptimumProjectorZeroColumn) {
  FeatureMatrix x = DATA(Feature, 4,
      1, 0, 1, 1, 1, 4, 0,
      2, 1, 0, 0, 0, 4, 0,
      3, 0, 1, 1, 1, 1, 0,
      4, 0, 1, 2, 2, 1, 0);

  ResponseVector y = DATA(Response, 4,
      0,
      0,
      1,
      1);

  FeatureVector actual = PPGLDAStrategy(0.1).optimize(x, GroupPartition(y));

  FeatureVector expected = DATA(Feature, 7,
      0, 0, 0, 0, 0, 1, 0);

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfZeroReturn) {
  FeatureMatrix x = DATA(Feature, 4,
      1, 0, 1, 1, 0, 1, 0, 1, 2, 3, 4, 5,
      1, 1, 0, 0, 0, 1, 0, 1, 2, 3, 4, 5,
      1, 1, 0, 1, 1, 0, 1, 0, 2, 3, 4, 5,
      1, 0, 1, 1, 1, 0, 1, 0, 2, 3, 4, 5);

  ResponseVector y = DATA(Response, 4,
      0,
      0,
      1,
      1);

  FeatureVector projector = DATA(Feature, 12,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);


  float actual = PPGLDAStrategy(0.5).index(x, GroupPartition(y), projector);

  ASSERT_EQ(0.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfOptimal1) {
  FeatureMatrix x = DATA(Feature, 30,
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

  ResponseVector y = DATA(Response, 30,
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

  FeatureVector projector = DATA(Feature, 5,
      -0.12823, -0.99174, 0.0, 0.0, 0.0);

  float actual = PPGLDAStrategy(0.5).index(x, GroupPartition(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfOptimal2) {
  FeatureMatrix x = DATA(Feature, 30,
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

  ResponseVector y = DATA(Response, 30,
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

  FeatureVector projector = DATA(Feature, 5,
      0.78481, 0.61974, 0.0, 0.0, 0.0);

  float actual = PPGLDAStrategy(0.5).index(x, GroupPartition(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfOptimal3) {
  FeatureMatrix x = DATA(Feature, 30,
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

  ResponseVector y = DATA(Response, 30,
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

  FeatureVector projector = DATA(Feature, 5,
      -0.66808,  0.74409,  0.0,  0.0,  0.0);

  float actual = PPGLDAStrategy(0.5).index(x, GroupPartition(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfSubptimal1) {
  FeatureMatrix x = DATA(Feature, 30,
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

  ResponseVector y = DATA(Response, 30,
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

  FeatureVector projector = DATA(Feature, 5,
      0, 0, 1, 1, 1);

  float actual = PPGLDAStrategy(0.5).index(x, GroupPartition(y), projector);

  ASSERT_NEAR(0.12597, actual, 0.00001);
}

TEST(Projector, PDAIndexLambdaOneHalfSubptimal2) {
  FeatureMatrix x = DATA(Feature, 30,
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

  ResponseVector y = DATA(Response, 30,
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

  FeatureVector projector = DATA(Feature, 5,
      -0.02965,  0.08452, -0.24243, -0.40089, -0.87892);

  float actual = PPGLDAStrategy(0.5).index(x, GroupPartition(y), projector);

  ASSERT_NEAR(0.0, actual, 0.000001);
}
