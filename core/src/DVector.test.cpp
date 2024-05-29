#include <gtest/gtest.h>

#include "DVector.hpp"

#define ASSERT_APPROX(a, b)    ASSERT_TRUE(a.isApprox(b, 0.00001)) << "Expected " << std::endl << a << std::endl << " to be approximate to " << std::endl << b
#define ASSERT_COLLINEAR(a, b) ASSERT_TRUE(collinear(a, b)) << "Expected columns of " << std::endl << a << std::endl << " to be collinear with its respective column of " << std::endl << b

TEST(DVector, OuterProductEqualVectors) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> b(3);
  b << 1.0, 2.0, 6.0;

  DMatrix<long double> actual = outer_product(a, b);

  DMatrix<long double> expected(3, 3);
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
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> b(3);
  b << 2.0, 3.0, 7.0;

  DMatrix<long double> actual = outer_product(a, b);

  DMatrix<long double> expected(3, 3);
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
  DVector<long double> a(3);
  a << 2.0, 3.0, 7.0;

  DVector<long double> b(3);
  b << 1.0, 2.0, 6.0;

  DMatrix<long double> actual = outer_product(a, b);

  DMatrix<long double> expected(3, 3);
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
  DVector<long double> a(3);
  a << 0.0, 0.0, 0.0;

  DVector<long double> b(3);
  b << 0.0, 0.0, 0.0;

  DMatrix<long double> actual = outer_product(a, b);

  DMatrix<long double> expected(3, 3);
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
  DVector<long double> a(1);
  a << 4.0;

  DVector<long double> b(1);
  b << 3.0;

  DMatrix<long double> actual = outer_product(a, b);
  DMatrix<long double> expected(1, 1);
  expected << 12.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, OuterSquareGeneric1) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DMatrix<long double> actual = outer_square(a);

  DMatrix<long double> expected(3, 3);
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
  DVector<long double> a(3);
  a << 2.0, 3.0, 7.0;

  DMatrix<long double> actual = outer_square(a);

  DMatrix<long double> expected(3, 3);
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
  DVector<long double> a(3);
  a << 0.0, 0.0, 0.0;

  DMatrix<long double> actual = outer_square(a);

  DMatrix<long double> expected(3, 3);
  expected <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, InnerProductEqualVectorsUnweighted) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> b(3);
  b << 1.0, 2.0, 6.0;

  DMatrix<long double> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  long double actual = inner_product(a, b, weights);
  long double expected = 41;

  ASSERT_EQ(expected, actual);
}

TEST(DVector, InnerProductEqualVectorsUnweightedImplicit) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> b(3);
  b << 1.0, 2.0, 6.0;

  long double actual = inner_product(a, b);
  long double expected = 41;

  ASSERT_EQ(expected, actual);
}

TEST(DVector, InnerProductEqualVectorsWeighted) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> b(3);
  b << 1.0, 2.0, 6.0;

  DMatrix<long double> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  long double actual = inner_product(a, b, weights);
  long double expected = 1681;

  ASSERT_EQ(expected, actual);
}

TEST(DVector, InnerProductZeroVectorsUnweighted) {
  DVector<long double> a(3);
  a << 0.0, 0.0, 0.0;

  DVector<long double> b(3);
  b << 0.0, 0.0, 0.0;

  DMatrix<long double> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  long double actual = inner_product(a, b, weights);
  long double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(DVector, InnerProductZeroVectorsUnweightedImplicit) {
  DVector<long double> a(3);
  a << 0.0, 0.0, 0.0;

  DVector<long double> b(3);
  b << 0.0, 0.0, 0.0;

  long double actual = inner_product(a, b);
  long double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(DVector, InnerProductZeroVectorsWeighted) {
  DVector<long double> a(3);
  a << 0.0, 0.0, 0.0;

  DVector<long double> b(3);
  b << 0.0, 0.0, 0.0;

  DMatrix<long double> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  long double actual = inner_product(a, b, weights);
  long double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(DVector, InnerProductDifferentVectorsUnweighted) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> b(3);
  b << 2.0, 3.0, 7.0;

  DMatrix<long double> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  long double actual = inner_product(a, b, weights);
  long double expected = 50;

  ASSERT_EQ(expected, actual);
}

TEST(DVector, InnerProductDifferentVectorsUnweightedImplicit) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> b(3);
  b << 2.0, 3.0, 7.0;

  long double actual = inner_product(a, b);
  long double expected = 50;

  ASSERT_EQ(expected, actual);
}

TEST(DVector, InnerProductDifferentVectorsWeighted) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> b(3);
  b << 2.0, 3.0, 7.0;

  DMatrix<long double> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  long double actual = inner_product(a, b, weights);
  long double expected = 2050;

  ASSERT_EQ(expected, actual);
}

TEST(DVector, InnerSquareGenericUnweighted) {
  DVector<long double> m(3);
  m << 1.0, 2.0, 6.0;

  long double actual = inner_square(m);
  long double expected = 41;

  ASSERT_EQ(expected, actual);
}

TEST(DVector, InnerSquareGenericWeightedIdentity) {
  DVector<long double> m(3);
  m << 1.0, 2.0, 6.0;

  DMatrix<long double> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  long double actual = inner_square(m, weights);
  long double expected = 41;

  ASSERT_EQ(expected, actual);
}

TEST(DVector, InnerSquareGenericWeighted) {
  DVector<long double> m(3);
  m << 1.0, 2.0, 6.0;

  DMatrix<long double> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  long double actual = inner_square(m, weights);
  long double expected = 1681;

  ASSERT_EQ(expected, actual);
}

TEST(DVector, CollinearCollinearSameDirection) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> b(3);
  b << 2.0, 4.0, 12.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(DVector, CollinearCollinearOppositeDirection) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> b(3);
  b << -1.0, -2.0, -6.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(DVector, CollinearNonCollinear) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> b(3);
  b << 2.0, 3.0, 7.0;

  ASSERT_FALSE(collinear(a, b));
}

TEST(DVector, AbsPositiveVector) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> actual = abs(a);
  DVector<long double> expected(3);
  expected << 1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, AbsNegativeVector) {
  DVector<long double> a(3);
  a << -1.0, -2.0, -6.0;

  DVector<long double> actual = abs(a);
  DVector<long double> expected(3);
  expected << 1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DVector, AbsMixedVec) {
  DVector<long double> a(3);
  a << -1.0, 2.0, -6.0;

  DVector<long double> actual = abs(a);
  DVector<long double> expected(3);
  expected << 1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}
