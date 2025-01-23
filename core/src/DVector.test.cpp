#include <gtest/gtest.h>

#include "DVector.hpp"

using namespace models::math;

#define ASSERT_APPROX(a, b)    ASSERT_TRUE(a.isApprox(b, 0.00001)) << "Expected " << std::endl << a << std::endl << " to be approximate to " << std::endl << b
#define ASSERT_COLLINEAR(a, b) ASSERT_TRUE(collinear(a, b)) << "Expected columns of " << std::endl << a << std::endl << " to be collinear with its respective column of " << std::endl << b

TEST(DVector, OuterProductEqualVectors) {
  DVector<double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<double> b(3);
  b << 1.0, 2.0, 6.0;

  DMatrix<double> actual = outer_product(a, b);

  DMatrix<double> expected(3, 3);
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
  DVector<double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<double> b(3);
  b << 2.0, 3.0, 7.0;

  DMatrix<double> actual = outer_product(a, b);

  DMatrix<double> expected(3, 3);
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
  DVector<double> a(3);
  a << 2.0, 3.0, 7.0;

  DVector<double> b(3);
  b << 1.0, 2.0, 6.0;

  DMatrix<double> actual = outer_product(a, b);

  DMatrix<double> expected(3, 3);
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
  DVector<double> a(3);
  a << 0.0, 0.0, 0.0;

  DVector<double> b(3);
  b << 0.0, 0.0, 0.0;

  DMatrix<double> actual = outer_product(a, b);

  DMatrix<double> expected(3, 3);
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
  DVector<double> a(1);
  a << 4.0;

  DVector<double> b(1);
  b << 3.0;

  DMatrix<double> actual = outer_product(a, b);
  DMatrix<double> expected(1, 1);
  expected << 12.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, OuterSquareGeneric1) {
  DVector<double> a(3);
  a << 1.0, 2.0, 6.0;

  DMatrix<double> actual = outer_square(a);

  DMatrix<double> expected(3, 3);
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
  DVector<double> a(3);
  a << 2.0, 3.0, 7.0;

  DMatrix<double> actual = outer_square(a);

  DMatrix<double> expected(3, 3);
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
  DVector<double> a(3);
  a << 0.0, 0.0, 0.0;

  DMatrix<double> actual = outer_square(a);

  DMatrix<double> expected(3, 3);
  expected <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, InnerProductEqualVectorsUnweightedImplicit) {
  DVector<double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<double> b(3);
  b << 1.0, 2.0, 6.0;

  double actual = inner_product(a, b);
  double expected = 41;

  ASSERT_EQ(expected, actual);
}

TEST(DVector, InnerProductZeroVectorsUnweightedImplicit) {
  DVector<double> a(3);
  a << 0.0, 0.0, 0.0;

  DVector<double> b(3);
  b << 0.0, 0.0, 0.0;

  double actual = inner_product(a, b);
  double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(DVector, InnerProductDifferentVectorsUnweightedImplicit) {
  DVector<double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<double> b(3);
  b << 2.0, 3.0, 7.0;

  double actual = inner_product(a, b);
  double expected = 50;

  ASSERT_EQ(expected, actual);
}

TEST(DVector, InnerSquareGenericUnweighted) {
  DVector<double> m(3);
  m << 1.0, 2.0, 6.0;

  double actual = inner_square(m);
  double expected = 41;

  ASSERT_EQ(expected, actual);
}

TEST(DVector, CollinearCollinearSameDirection) {
  DVector<double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<double> b(3);
  b << 2.0, 4.0, 12.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(DVector, CollinearCollinearOppositeDirection) {
  DVector<double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<double> b(3);
  b << -1.0, -2.0, -6.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(DVector, CollinearNonCollinear) {
  DVector<double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<double> b(3);
  b << 2.0, 3.0, 7.0;

  ASSERT_FALSE(collinear(a, b));
}

TEST(DVector, AbsPositiveVector) {
  DVector<double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<double> actual = abs(a);
  DVector<double> expected(3);
  expected << 1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, AbsNegativeVector) {
  DVector<double> a(3);
  a << -1.0, -2.0, -6.0;

  DVector<double> actual = abs(a);
  DVector<double> expected(3);
  expected << 1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, AbsMixedVec) {
  DVector<double> a(3);
  a << -1.0, 2.0, -6.0;

  DVector<double> actual = abs(a);
  DVector<double> expected(3);
  expected << 1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}
