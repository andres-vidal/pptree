#include <gtest/gtest.h>

#include "DVector.hpp"

#include "Macros.hpp"

using namespace models::math;

TEST(DVector, OuterProductEqualVectors) {
  DVector<float> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<float> b(3);
  b << 1.0, 2.0, 6.0;

  DMatrix<float> actual = outer_product(a, b);

  DMatrix<float> expected(3, 3);
  expected <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, OuterProductDifferentVectors1) {
  DVector<float> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<float> b(3);
  b << 2.0, 3.0, 7.0;

  DMatrix<float> actual = outer_product(a, b);

  DMatrix<float> expected(3, 3);
  expected <<
    2.0,  3.0,  7.0,
    4.0,  6.0,  14.0,
    12.0, 18.0, 42.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, OuterProductDifferentVectors2) {
  DVector<float> a(3);
  a << 2.0, 3.0, 7.0;

  DVector<float> b(3);
  b << 1.0, 2.0, 6.0;

  DMatrix<float> actual = outer_product(a, b);

  DMatrix<float> expected(3, 3);
  expected <<
    2.0,  4.0,  12.0,
    3.0,  6.0,  18.0,
    7.0,  14.0, 42.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, OuterProductZeroVectors) {
  DVector<float> a(3);
  a << 0.0, 0.0, 0.0;

  DVector<float> b(3);
  b << 0.0, 0.0, 0.0;

  DMatrix<float> actual = outer_product(a, b);

  DMatrix<float> expected(3, 3);
  expected <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, OuterProductGeneric1) {
  DVector<float> a(1);
  a << 4.0;

  DVector<float> b(1);
  b << 3.0;

  DMatrix<float> actual = outer_product(a, b);
  DMatrix<float> expected(1, 1);
  expected << 12.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, OuterSquareGeneric1) {
  DVector<float> a(3);
  a << 1.0, 2.0, 6.0;

  DMatrix<float> actual = outer_square(a);

  DMatrix<float> expected(3, 3);
  expected <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, OuterSquareGeneric2) {
  DVector<float> a(3);
  a << 2.0, 3.0, 7.0;

  DMatrix<float> actual = outer_square(a);

  DMatrix<float> expected(3, 3);
  expected <<
    4.0,  6.0,  14.0,
    6.0,  9.0,  21.0,
    14.0, 21.0, 49.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, OuterSquareZeroVector) {
  DVector<float> a(3);
  a << 0.0, 0.0, 0.0;

  DMatrix<float> actual = outer_square(a);

  DMatrix<float> expected(3, 3);
  expected <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, CollinearCollinearSameDirection) {
  DVector<float> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<float> b(3);
  b << 2.0, 4.0, 12.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(DVector, CollinearCollinearOppositeDirection) {
  DVector<float> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<float> b(3);
  b << -1.0, -2.0, -6.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(DVector, CollinearNonCollinear) {
  DVector<float> a(3);
  a << 0.0, 1.0, 0.0;

  DVector<float> b(3);
  b << 1.0, 0.0, 0.0;

  ASSERT_FALSE(collinear(a, b));
}

TEST(DVector, AbsPositiveVector) {
  DVector<float> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<float> actual = abs(a);
  DVector<float> expected(3);
  expected << 1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, AbsNegativeVector) {
  DVector<float> a(3);
  a << -1.0, -2.0, -6.0;

  DVector<float> actual = abs(a);
  DVector<float> expected(3);
  expected << 1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, AbsMixedVec) {
  DVector<float> a(3);
  a << -1.0, 2.0, -6.0;

  DVector<float> actual = abs(a);
  DVector<float> expected(3);
  expected << 1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}
