#include <gtest/gtest.h>

#include "PPStrategy.hpp"
#include "PPGLDAStrategy.hpp"

#include "Macros.hpp"

using namespace models::pp::strategy;
using namespace models::stats;

TEST(Projector, LDAOptimumProjectorTwoGroups1) {
  Data<float> x(10, 4);
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

  DataColumn<float> actual = PPGLDAStrategy<float, int>(0).optimize(x, DataSpec<int>(y));

  DataColumn<float> expected(4);
  expected <<
    -1, 0, 0, 0;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAOptimumProjectorTwoGroups2) {
  Data<float> x(10, 4);
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

  DataColumn<float> actual = PPGLDAStrategy<float, int>(0).optimize(x, DataSpec<int>(y));

  DataColumn<float> expected(4);
  expected <<
    0, 1, 0, 0;


  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAOptimumProjectorTwoGroups3) {
  Data<float> x(10, 4);
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

  DataColumn<float> actual = PPGLDAStrategy<float, int>(0).optimize(x, DataSpec<int>(y));

  DataColumn<float> expected(4);
  expected <<
    0, 0, -1, 0;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAOptimumProjectorTwoGroups4) {
  Data<float> x(10, 4);
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

  DataColumn<float> actual = PPGLDAStrategy<float, int>(0).optimize(x, DataSpec<int>(y));

  DataColumn<float> expected(4);
  expected <<
    2.0965219514666735e-15,
    4.4408920985006262e-16,
    -2.4980018054066002e-16,
    1;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAOptimumProjectorThreeGroups1) {
  Data<float> x(30, 5);
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

  DataColumn<float> actual = PPGLDAStrategy<float, int>(0).optimize(x, DataSpec<int>(y));

  DataColumn<float> expected(5);
  expected <<
    1,
    0,
    0,
    0,
    0;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, LDAIndexZeroReturn) {
  Data<float> x(4, 12);
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

  DataColumn<float> projector(12);
  projector <<
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  float actual = PPGLDAStrategy<float, int>(0).index(x, DataSpec<int>(y), projector);

  ASSERT_EQ(0.0, actual);
}

TEST(Projector, LDAIndexOptimal1) {
  Data<float> x(30, 5);
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

  DataColumn<float> projector(5);
  projector <<
    -0.12823, -0.99174, 0.0, 0.0, 0.0;


  float actual = PPGLDAStrategy<float, int>(0).index(x, DataSpec<int>(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, LDAIndexOptimal2) {
  Data<float> x(30, 5);
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

  DataColumn<float> projector(5);

  projector <<
    0.78481, 0.61974, 0.0, 0.0, 0.0;

  float actual = PPGLDAStrategy<float, int>(0).index(x, DataSpec<int>(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, LDAIndexOptimal3) {
  Data<float> x(30, 5);
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

  DataColumn<float> projector(5);

  projector <<
    -0.66808,  0.74409,  0.0,  0.0,  0.0;

  float actual = PPGLDAStrategy<float, int>(0).index(x, DataSpec<int>(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, LDAIndexSuboptimal1) {
  Data<float> x(30, 5);
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

  DataColumn<float> projector(5);

  projector <<
    0, 0, 1, 1, 1;

  float actual = PPGLDAStrategy<float, int>(0).index(x, DataSpec<int>(y), projector);

  ASSERT_NEAR(0.134985, actual, 0.00001);
}

TEST(Projector, LDAIndexSuboptimal2) {
  Data<float> x(30, 5);
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

  DataColumn<float> projector(5);

  projector <<
    -0.02965,  0.08452, -0.24243, -0.40089, -0.87892;

  float actual = PPGLDAStrategy<float, int>(0).index(x, DataSpec<int>(y), projector);

  ASSERT_NEAR(0.0, actual, 0.000001);
}

TEST(Projector, PDAOptimumProjectorLambdaOneHalfTwoGroups) {
  Data<float> x(4, 6);
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

  DataColumn<float> actual = PPGLDAStrategy<float, int>(0.5).optimize(x, DataSpec<int>(y));

  DataColumn<float> expected(6);
  expected <<
    0, 0, 0, 0, 0, 1;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, GLDAOptimumProjectorZeroColumn) {
  Data<float> x(4, 7);
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

  DataColumn<float> actual = PPGLDAStrategy<float, int>(0.1).optimize(x, DataSpec<int>(y));

  DataColumn<float> expected(7);
  expected <<
    0, 0, 0, 0, 0, 1, 0;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfZeroReturn) {
  Data<float> x(4, 12);
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

  DataColumn<float> projector(12);
  projector <<
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;


  float actual = PPGLDAStrategy<float, int>(0.5).index(x, DataSpec<int>(y), projector);

  ASSERT_EQ(0.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfOptimal1) {
  Data<float> x(30, 5);
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

  DataColumn<float> projector(5);
  projector <<
    -0.12823, -0.99174, 0.0, 0.0, 0.0;

  float actual = PPGLDAStrategy<float, int>(0.5).index(x, DataSpec<int>(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfOptimal2) {
  Data<float> x(30, 5);
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

  DataColumn<float> projector(5);

  projector <<
    0.78481, 0.61974, 0.0, 0.0, 0.0;

  float actual = PPGLDAStrategy<float, int>(0.5).index(x, DataSpec<int>(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfOptimal3) {
  Data<float> x(30, 5);
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

  DataColumn<float> projector(5);

  projector <<
    -0.66808,  0.74409,  0.0,  0.0,  0.0;

  float actual = PPGLDAStrategy<float, int>(0.5).index(x, DataSpec<int>(y), projector);

  ASSERT_FLOAT_EQ(1.0, actual);
}

TEST(Projector, PDAIndexLambdaOneHalfSubptimal1) {
  Data<float> x(30, 5);
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

  DataColumn<float> projector(5);

  projector <<
    0, 0, 1, 1, 1;

  float actual = PPGLDAStrategy<float, int>(0.5).index(x, DataSpec<int>(y), projector);

  ASSERT_NEAR(0.12597, actual, 0.00001);
}

TEST(Projector, PDAIndexLambdaOneHalfSubptimal2) {
  Data<float> x(30, 5);
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

  DataColumn<float> projector(5);

  projector <<
    -0.02965,  0.08452, -0.24243, -0.40089, -0.87892;

  float actual = PPGLDAStrategy<float, int>(0.5).index(x, DataSpec<int>(y), projector);

  ASSERT_NEAR(0.0, actual, 0.000001);
}
