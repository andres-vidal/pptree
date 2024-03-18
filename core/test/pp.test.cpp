#include <gtest/gtest.h>

#include "pp.hpp"

using namespace pp;
using namespace linalg;
using namespace stats;
using namespace Eigen;

#define ASSERT_COLLINEAR(a, b) ASSERT_TRUE(collinear(a, b)) << std::endl << "Expected vectors to be collinear: [" << a.transpose() << "] [" << b.transpose() << "]" << std::endl


TEST(PPLDAOptimumProjector, two_groups) {
  DMatrix<long double> data(10, 4);
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

  DVector<int> groups(10);
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

  DVector<long double> actual = lda_optimum_projector(data, groups, { 0, 1 });

  DVector<long double> expected(4);
  expected <<
    -1, 0, 0, 0;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(PPLDAOptimumProjector, two_groups2) {
  DMatrix<long double> data(10, 4);
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


  DVector<int> groups(10);
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

  DVector<long double> actual = lda_optimum_projector(data, groups, { 0, 1 });

  DVector<long double> expected(4);
  expected <<
    0, 1, 0, 0;


  ASSERT_COLLINEAR(expected, actual);
}

TEST(PPLDAOptimumProjector, two_groups3) {
  DMatrix<long double> data(10, 4);
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


  DVector<int> groups(10);
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

  DVector<long double> actual = lda_optimum_projector(data, groups, { 0, 1 });

  DVector<long double> expected(4);
  expected <<
    0, 0, -1, 0;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(PPLDAOptimumProjector, two_groups4) {
  DMatrix<long double> data(10, 4);
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

  DVector<int> groups(10);
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

  DVector<long double> actual = lda_optimum_projector(data, groups, { 0, 1 });

  DVector<long double> expected(4);
  expected <<
    2.0965219514666735e-15,
    4.4408920985006262e-16,
    -2.4980018054066002e-16,
    1;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(PPLDAOptimumProjector, three_groups) {
  DMatrix<long double> data(30, 5);
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

  DVector<int> groups(30);
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

  DVector<long double> actual = lda_optimum_projector(data, groups, { 0, 1, 2 });

  DVector<long double> expected(5);
  expected <<
    1,
    0,
    0,
    0,
    0;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(PPLDAIndex, zero_return) {
  DMatrix<long double> data(4, 12);
  data <<
    1, 0, 1, 1, 0, 1, 0, 1, 2, 3, 4, 5,
    1, 1, 0, 0, 0, 1, 0, 1, 2, 3, 4, 5,
    1, 1, 0, 1, 1, 0, 1, 0, 2, 3, 4, 5,
    1, 0, 1, 1, 1, 0, 1, 0, 2, 3, 4, 5;

  DVector<int> groups(4);
  groups <<
    0,
    0,
    1,
    1;

  DVector<long double> projector(12);
  projector <<
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  long double actual = lda_index(data, projector, groups, { 0, 1 });

  ASSERT_EQ(0.0, actual);
}

TEST(PPLDAIndex, optimal) {
  DMatrix<long double> data(30, 5);
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

  DVector<int> groups(30);
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

  DVector<long double> projector(5);
  projector <<
    -0.12823, -0.99174, 0.0, 0.0, 0.0;

  long double actual = lda_index(data, projector, groups, { 0, 1, 2 });

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(PPLDAIndex, optimal2) {
  DMatrix<long double> data(30, 5);
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

  DVector<int> groups(30);
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

  DVector<long double> projector(5);

  projector <<
    0.78481, 0.61974, 0.0, 0.0, 0.0;

  long double actual = lda_index(data, projector, groups, { 0, 1, 2 });

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(PPLDAIndex, optimal3) {
  DMatrix<long double> data(30, 5);
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

  DVector<int> groups(30);
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

  DVector<long double> projector(5);

  projector <<
    -0.66808,  0.74409,  0.0,  0.0,  0.0;

  long double actual = lda_index(data, projector, groups, { 0, 1, 2 });

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(PPLDAIndex, suboptimal) {
  DMatrix<long double> data(30, 5);
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

  DVector<int> groups(30);
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

  DVector<long double> projector(5);

  projector <<
    0, 0, 1, 1, 1;

  long double actual = lda_index(data, projector, groups, { 0, 1, 2 });

  ASSERT_NEAR(0.134985, actual, 0.00001);
}

TEST(PPLDAIndex, suboptimal2) {
  DMatrix<long double> data(30, 5);
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

  DVector<int> groups(30);
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

  DVector<long double> projector(5);

  projector <<
    -0.02965,  0.08452, -0.24243, -0.40089, -0.87892;

  long double actual = lda_index(data, projector, groups, { 0, 1, 2 });

  ASSERT_NEAR(0.0, actual, 0.000001);
}

TEST(PPPDAOptimumProjector, lambda_0_two_groups) {
  DMatrix<long double> data(10, 4);
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

  DVector<int> groups(10);
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

  DVector<long double> actual = pda_optimum_projector(data, groups, { 0, 1 }, 0);

  DVector<long double> expected(4);
  expected <<
    -1, 0, 0, 0;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(PPPDAOptimumProjector, lambda_0_two_groups2) {
  DMatrix<long double> data(10, 4);
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


  DVector<int> groups(10);
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

  DVector<long double> actual = pda_optimum_projector(data, groups, { 0, 1 }, 0);

  DVector<long double> expected(4);
  expected <<
    0, 1, 0, 0;


  ASSERT_COLLINEAR(expected, actual);
}

TEST(PPPDAOptimumProjector, lambda_0_two_groups3) {
  DMatrix<long double> data(10, 4);
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


  DVector<int> groups(10);
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

  DVector<long double> actual = pda_optimum_projector(data, groups, { 0, 1 }, 0);

  DVector<long double> expected(4);
  expected <<
    0, 0, -1, 0;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(PPPDAOptimumProjector, lambda_0_two_groups4) {
  DMatrix<long double> data(10, 4);
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

  DVector<int> groups(10);
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

  DVector<long double> actual = pda_optimum_projector(data, groups, { 0, 1 }, 0);

  DVector<long double> expected(4);
  expected <<
    2.0965219514666735e-15,
    4.4408920985006262e-16,
    -2.4980018054066002e-16,
    1;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(PPPDAOptimumProjector, lambda_0_three_groups) {
  DMatrix<long double> data(30, 5);
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

  DVector<int> groups(30);
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

  DVector<long double> actual = pda_optimum_projector(data, groups, { 0, 1, 2 }, 0);

  DVector<long double> expected(5);
  expected <<
    1,
    0,
    0,
    0,
    0;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(PPPDAOptimumProjector, lambda_onehalf_two_groups) {
  DMatrix<long double> data(4, 6);
  data <<
    1, 0, 1, 1, 1, 4,
    2, 1, 0, 0, 0, 4,
    3, 0, 1, 1, 1, 1,
    4, 0, 1, 2, 2, 1;

  DVector<int> groups(4);
  groups <<
    0,
    0,
    1,
    1;

  DVector<long double> actual = pda_optimum_projector(data, groups, { 0, 1 }, 0.5);

  DVector<long double> expected(6);
  expected <<
    0, 0, 0, 0, 0, 1;

  ASSERT_COLLINEAR(expected, actual);
}

TEST(PPProjectData, zero_projector) {
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

  Projector<long double> projector(5);
  projector <<
    0.0, 0.0, 0.0, 0.0, 0.0;

  DataColumn<long double> actual = project(data, projector);
  DataColumn<long double> expected = DataColumn<long double>::Zero(30);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(PPProjectData, generic_projector) {
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

  Projector<long double> projector(5);
  projector <<
    -0.02965,  0.08452, -0.24243, -0.40089, -0.87892;

  DataColumn<long double> actual = project(data, projector);
  DataColumn<long double> expected(30);
  expected <<
    -1.30946,
    -0.27208,
    -0.90857,
    -1.55189,
    -1.30946,
    -0.67297,
    -1.30946,
    -2.43081,
    -0.83143,
    -0.91540,
    -0.26206,
    -1.14098,
    -1.38341,
    -0.50449,
    -0.66295,
    -1.54187,
    -1.14781,
    -0.90538,
    -1.78430,
    -2.02673,
    -1.30627,
    -1.14781,
    -0.46961,
    -1.34853,
    -1.59096,
    -0.71204,
    -0.87050,
    -1.74942,
    -1.35536,
    -1.11293;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_TRUE(expected.isApprox(actual, 0.00001));
}

TEST(PPPProjectDataColumn, zero_projector) {
  DataColumn<long double> data(5);
  data <<
    1.0, 2.0, 3.0, 4.0, 5.0;

  Projector<long double> projector(5);
  projector <<
    0.0, 0.0, 0.0, 0.0, 0.0;

  long double result = project(data, projector);

  ASSERT_EQ(0, result);
}

TEST(PPPProjectDataColumn, generic_projector) {
  DataColumn<long double> data(5);
  data <<
    1.0, 2.0, 3.0, 4.0, 5.0;

  Projector<long double> projector(5);
  projector <<
    -0.02965,  0.08452, -0.24243, -0.40089, -0.87892;

  long double result = project(data, projector);

  ASSERT_NEAR(-6.58606, result, 0.00001);
}
