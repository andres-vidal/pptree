#include <gtest/gtest.h>

#include "PPStrategy.hpp"

using namespace models::pp;
using namespace models::pp::strategy;
using namespace models::stats;

#define ASSERT_COLLINEAR(a, b) ASSERT_TRUE(models::math::collinear(a.vector, b.vector)) << std::endl << "Expected vectors to be collinear: [" << a.vector.transpose() << "] [" << b.vector.transpose() << "]" << std::endl

TEST(Projector, LDAOptimumProjectorTwoGroups1) {
  Data<long double> data(10, 4);
  data <<
    1, 0, 1, 1,
    1, 1, 0, 0,
    1, 0, 0, 1,
    1, 1, 1, 1,
    4, 0, 0, 1,
    4, 0, 0, 2,
    4, 0, 0, 3,
    4, 1, 0, 1,
    4, 0, 1, 1,
    4, 0, 1, 2;

  DataColumn<int> groups(10);
  groups <<
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1;

  Projector<long double> actual = GLDAStrategy<long double, int>(0).optimize(data, groups, { 0, 1 });

  Projector<long double> expected({ -1, 0, 0, 0 });

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAOptimumProjectorTwoGroups2) {
  Data<long double> data(10, 4);
  data <<
    0, 1, 1, 1,
    1, 1, 0, 0,
    0, 1, 0, 1,
    1, 1, 1, 1,
    0, 4, 0, 1,
    0, 4, 0, 2,
    0, 4, 0, 3,
    1, 4, 0, 1,
    0, 4, 1, 1,
    0, 4, 1, 2;


  DataColumn<int> groups(10);
  groups <<
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1;

  Projector<long double> actual = GLDAStrategy<long double, int>(0).optimize(data, groups, { 0, 1 });

  Projector<long double> expected({ 0, 1, 0, 0 });

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAOptimumProjectorTwoGroups3) {
  Data<long double> data(10, 4);
  data <<
    0, 1, 1, 1,
    1, 0, 1, 0,
    0, 0, 1, 1,
    1, 1, 1, 1,
    0, 0, 4, 1,
    0, 0, 4, 2,
    0, 0, 4, 3,
    1, 0, 4, 1,
    0, 1, 4, 1,
    0, 1, 4, 2;


  DataColumn<int> groups(10);
  groups <<
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1;

  Projector<long double> actual = GLDAStrategy<long double, int>(0).optimize(data, groups, { 0, 1 });

  Projector<long double> expected({ 0, 0, -1, 0 });


  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAOptimumProjectorTwoGroups4) {
  Data<long double> data(10, 4);
  data <<
    0, 1, 1, 1,
    1, 0, 0, 1,
    0, 0, 1, 1,
    1, 1, 1, 1,
    0, 0, 1, 4,
    0, 0, 2, 4,
    0, 0, 3, 4,
    1, 0, 1, 4,
    0, 1, 1, 4,
    0, 1, 2, 4;

  DataColumn<int> groups(10);
  groups <<
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1;

  Projector<long double> actual = GLDAStrategy<long double, int>(0).optimize(data, groups, { 0, 1 });

  Projector<long double> expected({
    2.0965219514666735e-15,
    4.4408920985006262e-16,
    -2.4980018054066002e-16,
    1
  });

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAOptimumProjectorThreeGroups1) {
  Data<long double> data(30, 5);
  data <<
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
    9, 8, 1, 1, 1;

  DataColumn<int> groups(30);
  groups <<
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
    2;

  Projector<long double> actual = GLDAStrategy<long double, int>(0).optimize(data, groups, { 0, 1, 2 });

  Projector<long double> expected({
    1,
    0,
    0,
    0,
    0
  });

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAIndexZeroReturn) {
  Data<long double> data(4, 12);
  data <<
    1, 0, 1, 1, 0, 1, 0, 1, 2, 3, 4, 5,
    1, 1, 0, 0, 0, 1, 0, 1, 2, 3, 4, 5,
    1, 1, 0, 1, 1, 0, 1, 0, 2, 3, 4, 5,
    1, 0, 1, 1, 1, 0, 1, 0, 2, 3, 4, 5;

  DataColumn<int> groups(4);
  groups <<
    0,
    0,
    1,
    1;

  Projector<long double> projector({  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });

  long double actual = GLDAStrategy<long double, int>(0).index(data, projector, groups, { 0, 1 });

  ASSERT_EQ(0.0, actual);
}

TEST(Projector, LDAIndexOptimal1) {
  Data<long double> data(30, 5);
  data <<
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
    9, 8, 1, 1, 1;

  DataColumn<int> groups(30);
  groups <<
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
    2;

  Projector<long double> projector({ -0.12823, -0.99174, 0.0, 0.0, 0.0 });

  long double actual = GLDAStrategy<long double, int>(0).index(data, projector, groups, { 0, 1, 2 });

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(Projector, LDAIndexOptimal2) {
  Data<long double> data(30, 5);
  data <<
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
    9, 8, 1, 1, 1;

  DataColumn<int> groups(30);
  groups <<
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
    2;

  Projector<long double> projector({ 0.78481, 0.61974, 0.0, 0.0, 0.0 });

  long double actual = GLDAStrategy<long double, int>(0).index(data, projector, groups, { 0, 1, 2 });

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(Projector, LDAIndexOptimal3) {
  Data<long double> data(30, 5);
  data <<
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
    9, 8, 1, 1, 1;

  DataColumn<int> groups(30);
  groups <<
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
    2;

  Projector<long double> projector({ -0.66808,  0.74409,  0.0,  0.0,  0.0 });

  long double actual = GLDAStrategy<long double, int>(0).index(data, projector, groups, { 0, 1, 2 });

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(Projector, LDAIndexSuboptimal1) {
  Data<long double> data(30, 5);
  data <<
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
    9, 8, 1, 1, 1;

  DataColumn<int> groups(30);
  groups <<
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
    2;

  Projector<long double> projector({ 0, 0, 1, 1, 1 });

  long double actual = GLDAStrategy<long double, int>(0).index(data, projector, groups, { 0, 1, 2 });

  ASSERT_NEAR(0.134985, actual, 0.00001);
}

TEST(Projector, LDAIndexSuboptimal2) {
  Data<long double> data(30, 5);
  data <<
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
    9, 8, 1, 1, 1;

  DataColumn<int> groups(30);
  groups <<
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
    2;

  Projector<long double> projector({ -0.02965,  0.08452, -0.24243, -0.40089, -0.87892 });

  long double actual = GLDAStrategy<long double, int>(0).index(data, projector, groups, { 0, 1, 2 });

  ASSERT_NEAR(0.0, actual, 0.000001);
}

TEST(Projector, PDAOptimumProjectorLambdaOneHalfTwoGroups) {
  Data<long double> data(4, 6);
  data <<
    1, 0, 1, 1, 1, 4,
    2, 1, 0, 0, 0, 4,
    3, 0, 1, 1, 1, 1,
    4, 0, 1, 2, 2, 1;

  DataColumn<int> groups(4);
  groups <<
    0,
    0,
    1,
    1;

  Projector<long double> actual = GLDAStrategy<long double, int>(0.5).optimize(data, groups, { 0, 1 });

  Projector<long double> expected({  0, 0, 0, 0, 0, 1 });

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, GLDAOptimumProjectorZeroColumn) {
  Data<long double> data(4, 7);
  data <<
    1, 0, 1, 1, 1, 4, 0,
    2, 1, 0, 0, 0, 4, 0,
    3, 0, 1, 1, 1, 1, 0,
    4, 0, 1, 2, 2, 1, 0;

  DataColumn<int> groups(4);
  groups <<
    0,
    0,
    1,
    1;

  Projector<long double> actual = GLDAStrategy<long double, int>(0.1).optimize(data, groups, { 0, 1 });

  Projector<long double> expected({ 0, 0, 0, 0, 0, 1, 0 });

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfZeroReturn) {
  Data<long double> data(4, 12);
  data <<
    1, 0, 1, 1, 0, 1, 0, 1, 2, 3, 4, 5,
    1, 1, 0, 0, 0, 1, 0, 1, 2, 3, 4, 5,
    1, 1, 0, 1, 1, 0, 1, 0, 2, 3, 4, 5,
    1, 0, 1, 1, 1, 0, 1, 0, 2, 3, 4, 5;

  DataColumn<int> groups(4);
  groups <<
    0,
    0,
    1,
    1;

  Projector<long double> projector({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });

  long double actual = GLDAStrategy<long double, int>(0.5).index(data, projector, groups, { 0, 1 });

  ASSERT_EQ(0.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfOptimal1) {
  Data<long double> data(30, 5);
  data <<
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
    9, 8, 1, 1, 1;

  DataColumn<int> groups(30);
  groups <<
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
    2;

  Projector<long double> projector({   -0.12823, -0.99174, 0.0, 0.0, 0.0 });

  long double actual = GLDAStrategy<long double, int>(0.5).index(data, projector, groups, { 0, 1, 2 });

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfOptimal2) {
  Data<long double> data(30, 5);
  data <<
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
    9, 8, 1, 1, 1;

  DataColumn<int> groups(30);
  groups <<
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
    2;

  Projector<long double> projector({ 0.78481, 0.61974, 0.0, 0.0, 0.0 });

  long double actual = GLDAStrategy<long double, int>(0.5).index(data, projector, groups, { 0, 1, 2 });

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfOptimal3) {
  Data<long double> data(30, 5);
  data <<
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
    9, 8, 1, 1, 1;

  DataColumn<int> groups(30);
  groups <<
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
    2;

  long double actual = GLDAStrategy<long double, int>(0.5).index(data, Projector<long double>({ -0.66808,  0.74409,  0.0,  0.0,  0.0 }), groups, { 0, 1, 2 });

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfSubptimal1) {
  Data<long double> data(30, 5);
  data <<
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
    9, 8, 1, 1, 1;

  DataColumn<int> groups(30);
  groups <<
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
    2;

  Projector<long double> projector({ 0, 0, 1, 1, 1 });

  long double actual = GLDAStrategy<long double, int>(0.5).index(data, projector, groups, { 0, 1, 2 });

  ASSERT_NEAR(0.12597, actual, 0.00001);
}

TEST(Projector, PDAIndexLambdaOneHalfSubptimal2) {
  Data<long double> data(30, 5);
  data <<
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
    9, 8, 1, 1, 1;

  DataColumn<int> groups(30);
  groups <<
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
    2;

  Projector<long double> projector({ -0.02965,  0.08452, -0.24243, -0.40089, -0.87892 });

  long double actual = GLDAStrategy<long double, int>(0.5).index(data, projector, groups, { 0, 1, 2 });

  ASSERT_NEAR(0.0, actual, 0.000001);
}
