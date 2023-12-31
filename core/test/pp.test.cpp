#include <gtest/gtest.h>

#include "pp.hpp"

TEST(PPLDAOptimumProjector, two_groups) {
  DMatrix<double> data(10, 4);
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

  DVector<unsigned short> groups(10);
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

  DVector<double> actual = pp::lda_optimum_projector(data, groups, 2);

  DVector<double> expected(4);
  expected <<
    -1, 0, 0, 0;

  ASSERT_TRUE(actual.isApprox(expected, 0.0001));
}

TEST(PPLDAOptimumProjector, two_groups2) {
  DMatrix<double> data(10, 4);
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


  DVector<unsigned short> groups(10);
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

  DVector<double> actual = pp::lda_optimum_projector(data, groups, 2);

  DVector<double> expected(4);
  expected <<
    0, 1, 0, 0;


  ASSERT_TRUE(actual.isApprox(expected, 0.0001));
}

TEST(PPLDAOptimumProjector, two_groups3) {
  DMatrix<double> data(10, 4);
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


  DVector<unsigned short> groups(10);
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

  DVector<double> actual = pp::lda_optimum_projector(data, groups, 2);

  DVector<double> expected(4);
  expected <<
    0, 0, -1, 0;

  ASSERT_TRUE(actual.isApprox(expected, 0.0001) || actual.isApprox(-expected, 0.0001));
}

TEST(PPLDAOptimumProjector, two_groups4) {
  DMatrix<double> data(10, 4);
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

  DVector<unsigned short> groups(10);
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

  DVector<double> actual = pp::lda_optimum_projector(data, groups, 2);

  DVector<double> expected(4);
  expected <<
    0, 0, 0, -1;

  ASSERT_TRUE(actual.isApprox(expected, 0.0001) || actual.isApprox(-expected, 0.0001));
}

TEST(PPLDAOptimumProjector, three_groups) {
  DMatrix<double> data(30, 5);
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

  DVector<unsigned short> groups(30);
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

  DVector<double> actual = pp::lda_optimum_projector(data, groups, 3);

  DVector<double> expected1(5);
  expected1 <<
    -0.12823, -0.99174, 0, 0, 0;

  DVector<double> expected2(5);
  expected2 <<
    0.64183, -0.76684, 0, 0, 0;

  std::cout << actual << std::endl;

  ASSERT_TRUE(
    actual.isApprox(expected1, 0.0001) ||
    actual.isApprox(-expected1, 0.0001) ||
    actual.isApprox(expected2, 0.0001) ||
    actual.isApprox(-expected2, 0.0001)
    );
}

TEST(PPLDAIndex, zero_return) {
  DMatrix<double> data(4, 12);
  data <<
    1, 0, 1, 1, 0, 1, 0, 1, 2, 3, 4, 5,
    1, 1, 0, 0, 0, 1, 0, 1, 2, 3, 4, 5,
    1, 1, 0, 1, 1, 0, 1, 0, 2, 3, 4, 5,
    1, 0, 1, 1, 1, 0, 1, 0, 2, 3, 4, 5;

  DVector<unsigned short> groups(4);
  groups <<
    0,
    0,
    1,
    1;

  DVector<double> projection_vector(12);
  projection_vector <<
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  double actual = pp::lda_index(data, projection_vector, groups, 2);

  ASSERT_EQ(0.0, actual);
}

TEST(PPLDAIndex, optimal) {
  DMatrix<double> data(30, 5);
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

  DVector<unsigned short> groups(30);
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

  DVector<double> projection_vector(5);
  projection_vector <<
    -0.12823, -0.99174, 0.0, 0.0, 0.0;

  double actual = pp::lda_index(data, projection_vector, groups, 3);

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(PPLDAIndex, optimal2) {
  DMatrix<double> data(30, 5);
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

  DVector<unsigned short> groups(30);
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

  DVector<double> projection_vector(5);

  projection_vector <<
    0.78481, 0.61974, 0.0, 0.0, 0.0;

  double actual = pp::lda_index(data, projection_vector, groups, 3);

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(PPLDAIndex, optimal3) {
  DMatrix<double> data(30, 5);
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

  DVector<unsigned short> groups(30);
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

  DVector<double> projection_vector(5);

  projection_vector <<
    -0.66808,  0.74409,  0.0,  0.0,  0.0;

  double actual = pp::lda_index(data, projection_vector, groups, 3);

  ASSERT_DOUBLE_EQ(1.0, actual);
}

TEST(PPLDAIndex, suboptimal) {
  DMatrix<double> data(30, 5);
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

  DVector<unsigned short> groups(30);
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

  DVector<double> projection_vector(5);

  projection_vector <<
    0, 0, 1, 1, 1;

  double actual = pp::lda_index(data, projection_vector, groups, 3);

  ASSERT_NEAR(0.134985, actual, 0.00001);
}

TEST(PPLDAIndex, suboptimal2) {
  DMatrix<double> data(30, 5);
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

  DVector<unsigned short> groups(30);
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

  DVector<double> projection_vector(5);

  projection_vector <<
    -0.02965,  0.08452, -0.24243, -0.40089, -0.87892;

  double actual = pp::lda_index(data, projection_vector, groups, 3);

  ASSERT_NEAR(0.0, actual, 0.000001);
}
