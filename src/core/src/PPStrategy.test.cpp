#include <gtest/gtest.h>

#include "PPStrategy.hpp"

using namespace models::pp::strategy;
using namespace models::stats;

#define ASSERT_COLLINEAR(a, b) ASSERT_TRUE(models::math::collinear(a, b)) << std::endl << "Expected vectors to be collinear: [" << a.transpose() << "] [" << b.transpose() << "]" << std::endl

TEST(Projector, LDAOptimumProjectorTwoGroups1) {
  Data<double> x(10, 4);
  x <<
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

  DataColumn<int> y(10);
  y <<
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

  SortedDataSpec<double, int> data(x, y);

  DataColumn<double> actual = GLDAStrategy<double, int>(0).optimize(data);

  DataColumn<double> expected(4);
  expected <<
    -1, 0, 0, 0;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAOptimumProjectorTwoGroups2) {
  Data<double> x(10, 4);
  x <<
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


  DataColumn<int> y(10);
  y <<
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

  SortedDataSpec<double, int> data(x, y);

  DataColumn<double> actual = GLDAStrategy<double, int>(0).optimize(data);

  DataColumn<double> expected(4);
  expected <<
    0, 1, 0, 0;


  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAOptimumProjectorTwoGroups3) {
  Data<double> x(10, 4);
  x <<
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


  DataColumn<int> y(10);
  y <<
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

  SortedDataSpec<double, int> data(x, y);

  DataColumn<double> actual = GLDAStrategy<double, int>(0).optimize(data);

  DataColumn<double> expected(4);
  expected <<
    0, 0, -1, 0;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAOptimumProjectorTwoGroups4) {
  Data<double> x(10, 4);
  x <<
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

  DataColumn<int> y(10);
  y <<
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

  SortedDataSpec<double, int> data(x, y);

  DataColumn<double> actual = GLDAStrategy<double, int>(0).optimize(data);

  DataColumn<double> expected(4);
  expected <<
    2.0965219514666735e-15,
    4.4408920985006262e-16,
    -2.4980018054066002e-16,
    1;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAOptimumProjectorThreeGroups1) {
  Data<double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  SortedDataSpec<double, int> data(x, y);

  DataColumn<double> actual = GLDAStrategy<double, int>(0).optimize(data);

  DataColumn<double> expected(5);
  expected <<
    1,
    0,
    0,
    0,
    0;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAIndexZeroReturn) {
  Data<double> x(4, 12);
  x <<
    1, 0, 1, 1, 0, 1, 0, 1, 2, 3, 4, 5,
    1, 1, 0, 0, 0, 1, 0, 1, 2, 3, 4, 5,
    1, 1, 0, 1, 1, 0, 1, 0, 2, 3, 4, 5,
    1, 0, 1, 1, 1, 0, 1, 0, 2, 3, 4, 5;

  DataColumn<int> y(4);
  y <<
    0,
    0,
    1,
    1;

  DataColumn<double> projector(12);
  projector <<
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  SortedDataSpec<double, int> data(x, y);

  double actual = GLDAStrategy<double, int>(0).index(data, projector);

  ASSERT_EQ(0.0, actual);
}

TEST(Projector, LDAIndexOptimal1) {
  Data<double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataColumn<double> projector(5);
  projector <<
    -0.12823, -0.99174, 0.0, 0.0, 0.0;


  SortedDataSpec<double, int> data(x, y);

  double actual = GLDAStrategy<double, int>(0).index(data, projector);

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(Projector, LDAIndexOptimal2) {
  Data<double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataColumn<double> projector(5);

  projector <<
    0.78481, 0.61974, 0.0, 0.0, 0.0;

  SortedDataSpec<double, int> data(x, y);

  double actual = GLDAStrategy<double, int>(0).index(data, projector);

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(Projector, LDAIndexOptimal3) {
  Data<double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataColumn<double> projector(5);

  projector <<
    -0.66808,  0.74409,  0.0,  0.0,  0.0;

  SortedDataSpec<double, int> data(x, y);

  double actual = GLDAStrategy<double, int>(0).index(data, projector);

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(Projector, LDAIndexSuboptimal1) {
  Data<double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataColumn<double> projector(5);

  projector <<
    0, 0, 1, 1, 1;

  SortedDataSpec<double, int> data(x, y);

  double actual = GLDAStrategy<double, int>(0).index(data, projector);

  ASSERT_NEAR(0.134985, actual, 0.00001);
}

TEST(Projector, LDAIndexSuboptimal2) {
  Data<double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataColumn<double> projector(5);

  projector <<
    -0.02965,  0.08452, -0.24243, -0.40089, -0.87892;

  SortedDataSpec<double, int> data(x, y);

  double actual = GLDAStrategy<double, int>(0).index(data, projector);

  ASSERT_NEAR(0.0, actual, 0.000001);
}

TEST(Projector, PDAOptimumProjectorLambdaOneHalfTwoGroups) {
  Data<double> x(4, 6);
  x <<
    1, 0, 1, 1, 1, 4,
    2, 1, 0, 0, 0, 4,
    3, 0, 1, 1, 1, 1,
    4, 0, 1, 2, 2, 1;

  DataColumn<int> y(4);
  y <<
    0,
    0,
    1,
    1;

  SortedDataSpec<double, int> data(x, y);

  DataColumn<double> actual = GLDAStrategy<double, int>(0.5).optimize(data);

  DataColumn<double> expected(6);
  expected <<
    0, 0, 0, 0, 0, 1;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, GLDAOptimumProjectorZeroColumn) {
  Data<double> x(4, 7);
  x <<
    1, 0, 1, 1, 1, 4, 0,
    2, 1, 0, 0, 0, 4, 0,
    3, 0, 1, 1, 1, 1, 0,
    4, 0, 1, 2, 2, 1, 0;

  DataColumn<int> y(4);
  y <<
    0,
    0,
    1,
    1;

  SortedDataSpec<double, int> data(x, y);

  DataColumn<double> actual = GLDAStrategy<double, int>(0.1).optimize(data);

  DataColumn<double> expected(7);
  expected <<
    0, 0, 0, 0, 0, 1, 0;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfZeroReturn) {
  Data<double> x(4, 12);
  x <<
    1, 0, 1, 1, 0, 1, 0, 1, 2, 3, 4, 5,
    1, 1, 0, 0, 0, 1, 0, 1, 2, 3, 4, 5,
    1, 1, 0, 1, 1, 0, 1, 0, 2, 3, 4, 5,
    1, 0, 1, 1, 1, 0, 1, 0, 2, 3, 4, 5;

  DataColumn<int> y(4);
  y <<
    0,
    0,
    1,
    1;

  DataColumn<double> projector(12);
  projector <<
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  SortedDataSpec<double, int> data(x, y);

  double actual = GLDAStrategy<double, int>(0.5).index(data, projector);

  ASSERT_EQ(0.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfOptimal1) {
  Data<double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataColumn<double> projector(5);
  projector <<
    -0.12823, -0.99174, 0.0, 0.0, 0.0;

  SortedDataSpec<double, int> data(x, y);

  double actual = GLDAStrategy<double, int>(0.5).index(data, projector);

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfOptimal2) {
  Data<double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataColumn<double> projector(5);

  projector <<
    0.78481, 0.61974, 0.0, 0.0, 0.0;

  SortedDataSpec<double, int> data(x, y);

  double actual = GLDAStrategy<double, int>(0.5).index(data, projector);

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfOptimal3) {
  Data<double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataColumn<double> projector(5);

  projector <<
    -0.66808,  0.74409,  0.0,  0.0,  0.0;

  SortedDataSpec<double, int> data(x, y);

  double actual = GLDAStrategy<double, int>(0.5).index(data, projector);

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfSubptimal1) {
  Data<double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataColumn<double> projector(5);

  projector <<
    0, 0, 1, 1, 1;

  SortedDataSpec<double, int> data(x, y);

  double actual = GLDAStrategy<double, int>(0.5).index(data, projector);

  ASSERT_NEAR(0.12597, actual, 0.00001);
}

TEST(Projector, PDAIndexLambdaOneHalfSubptimal2) {
  Data<double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataColumn<double> projector(5);

  projector <<
    -0.02965,  0.08452, -0.24243, -0.40089, -0.87892;

  SortedDataSpec<double, int> data(x, y);

  double actual = GLDAStrategy<double, int>(0.5).index(data, projector);

  ASSERT_NEAR(0.0, actual, 0.000001);
}