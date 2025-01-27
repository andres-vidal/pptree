#include <gtest/gtest.h>

#include "DMatrix.hpp"

using namespace models::math;

#define ASSERT_APPROX(a, b)    ASSERT_TRUE(a.isApprox(b, 0.00001)) << "Expected " << std::endl << a << std::endl << " to be approximate to " << std::endl << b
#define ASSERT_COLLINEAR(a, b) ASSERT_TRUE(collinear(a, b)) << "Expected columns of " << std::endl << a << std::endl << " to be collinear with its respective column of " << std::endl << b

TEST(DMatrix, InnerProductEqualMatricesUnweighted) {
  DMatrix<double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<double> b(3, 3);
  b <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<double> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<double> actual = inner_product(a, b, weights);
  DMatrix<double> expected(3, 3);
  expected <<
    14.0, 20.0, 44.0,
    20.0, 29.0, 65.0,
    44.0, 65.0, 149.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, InnerProductEqualMatricesUnweightedImplicit) {
  DMatrix<double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<double> b(3, 3);
  b <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;


  DMatrix<double> actual = inner_product(a, b);
  DMatrix<double> expected(3, 3);
  expected <<
    14.0, 20.0, 44.0,
    20.0, 29.0, 65.0,
    44.0, 65.0, 149.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, InnerProductEqualMatricesWeighted) {
  DMatrix<double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<double> b(3, 3);
  b <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<double> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  DMatrix<double> actual = inner_product(a, b, weights);
  DMatrix<double> expected(3, 3);
  expected <<
    529.0,  736.0,  1564.0,
    736.0,  1024.0, 2176.0,
    1564.0, 2176.0, 4624.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, InnerProductZeroMatricesUnweighted) {
  DMatrix<double> a(3, 3);
  a <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<double> b(3, 3);
  b <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<double> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<double> actual = inner_product(a, b, weights);
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

TEST(DMatrix, InnerProductZeroMatricesUnweightedImplicit) {
  DMatrix<double> a(3, 3);
  a <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<double> b(3, 3);
  b <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<double> actual = inner_product(a, b);
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

TEST(DMatrix, InnerProductZeroMatricesWeighted) {
  DMatrix<double> a(3, 3);
  a <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<double> b(3, 3);
  b <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<double> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  DMatrix<double> actual = inner_product(a, b, weights);
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

TEST(DMatrix, InnerProductDifferentMatricesUnweighted) {
  DMatrix<double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<double> b(3, 3);
  b <<
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0,
    4.0, 5.0, 9.0;

  DMatrix<double> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<double> actual = inner_product(a, b, weights);
  DMatrix<double> expected(3, 3);
  expected <<
    20.0, 26.0, 50.0,
    29.0, 38.0, 74.0,
    65.0, 86.0, 170.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, InnerProductDifferentMatricesUnweightedImplicit) {
  DMatrix<double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<double> b(3, 3);
  b <<
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0,
    4.0, 5.0, 9.0;

  DMatrix<double> actual = inner_product(a, b);
  DMatrix<double> expected(3, 3);
  expected <<
    20.0, 26.0, 50.0,
    29.0, 38.0, 74.0,
    65.0, 86.0, 170.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, InnerProductDifferentMatricesWeighted) {
  DMatrix<double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<double> b(3, 3);
  b <<
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0,
    4.0, 5.0, 9.0;

  DMatrix<double> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  DMatrix<double> actual = inner_product(a, b, weights);
  DMatrix<double> expected(3, 3);
  expected <<
    736.0,  943.0,  1771.0,
    1024.0, 1312.0, 2464.0,
    2176.0, 2788.0, 5236.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, InnerSquareGenericUnweighted) {
  DMatrix<double> m(3, 3);
  m <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<double> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<double> actual = inner_square(m, weights);
  DMatrix<double> expected(3, 3);
  expected <<
    14.0, 20.0, 44.0,
    20.0, 29.0, 65.0,
    44.0, 65.0, 149.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, InnerSquareGenericUnweightedImplicit) {
  DMatrix<double> m(3, 3);
  m <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<double> actual = inner_square(m);
  DMatrix<double> expected(3, 3);
  expected <<
    14.0, 20.0, 44.0,
    20.0, 29.0, 65.0,
    44.0, 65.0, 149.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, InnerSquareGenericWeighted) {
  DMatrix<double> m(3, 3);
  m <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<double> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  DMatrix<double> actual = inner_square(m, weights);
  DMatrix<double> expected(3, 3);
  expected <<
    529.0,  736.0,  1564.0,
    736.0,  1024.0, 2176.0,
    1564.0, 2176.0, 4624.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, DeterminantGenericPositiveDeterminant) {
  DMatrix<double> m(3, 3);
  m <<
    6.0, 1.0, 4.0,
    4.0, 8.0, 4.0,
    6.0, 3.0, 5.0;

  double actual = determinant(m);
  double expected = 28.0;

  ASSERT_DOUBLE_EQ(expected, actual);
}

TEST(DMatrix, DeterminantGenericNegativeDeterminant) {
  DMatrix<double> m(3, 3);
  m <<
    6.0, 1.0, 4.0,
    4.0, 8.0, 4.0,
    8.0, 3.0, 5.0;

  double actual = determinant(m);
  double expected = -28.0;

  ASSERT_DOUBLE_EQ(expected, actual);
}

TEST(DMatrix, DeterminantZeroMatrix) {
  DMatrix<double> m(3, 3);
  m <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  double actual = determinant(m);
  double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, DeterminantSingularMatrix) {
  DMatrix<double> m(3, 3);
  m <<
    1.0, 2.0, 6.0,
    2.0, 4.0, 7.0,
    3.0, 6.0, 8.0;

  double actual = determinant(m);
  double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, SolveIdentityIdentity) {
  DMatrix<double> l(3, 3);
  l <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<double> r(3, 3);
  r <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<double> actual = solve(l, r);
  DMatrix<double> expected(3, 3);
  expected <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, SolveGenericIdentity) {
  DMatrix<double> l(3, 3);
  l <<
    0.0, 1.0, 2.0,
    1.0, 2.0, 3.0,
    3.0, 1.0, 1.0;

  DMatrix<double> r(3, 3);
  r <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<double> actual = solve(l, r);
  DMatrix<double> expected(3, 3);
  expected <<
    0.5,  -0.5,   0.5,
    -4.0,  3.0,  -1.0,
    2.5,  -1.5,   0.5;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_APPROX(expected, actual);
}

TEST(DMatrix, EigenIdentity) {
  DMatrix<double> m(3, 3);
  m <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  auto [actual_values, actual_vectors] = eigen(m);

  DVector<double> expected_values(3);
  expected_values <<
    1.0, 1.0, 1.0;

  DMatrix<double> expected_vectors(3, 3);
  expected_vectors <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  ASSERT_EQ(expected_values.size(), actual_values.size());
  ASSERT_EQ(expected_values.rows(), actual_values.rows());
  ASSERT_EQ(expected_values.cols(), actual_values.cols());
  ASSERT_EQ(expected_values, actual_values);

  ASSERT_EQ(expected_vectors.size(), actual_vectors.size());
  ASSERT_EQ(expected_vectors.rows(), actual_vectors.rows());
  ASSERT_EQ(expected_vectors.cols(), actual_vectors.cols());
  ASSERT_EQ(expected_vectors, actual_vectors);
}

TEST(DMatrix, EigenDiagonal) {
  DMatrix<double> m(3, 3);
  m <<
    1.0, 0.0, 0.0,
    0.0, 2.0, 0.0,
    0.0, 0.0, 3.0;

  auto [actual_values, actual_vectors] = eigen(m);

  DVector<double> expected_values(3);
  expected_values <<
    1.0, 2.0, 3.0;

  DMatrix<double> expected_vectors(3, 3);
  expected_vectors <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  ASSERT_EQ(expected_values.size(), actual_values.size());
  ASSERT_EQ(expected_values.rows(), actual_values.rows());
  ASSERT_EQ(expected_values.cols(), actual_values.cols());
  ASSERT_EQ(expected_values, actual_values);

  ASSERT_EQ(expected_vectors.size(), actual_vectors.size());
  ASSERT_EQ(expected_vectors.rows(), actual_vectors.rows());
  ASSERT_EQ(expected_vectors.cols(), actual_vectors.cols());
  ASSERT_EQ(expected_vectors, actual_vectors);
}

TEST(DMatrix, EigenSymmetricRealNonNegativeEigenvalues) {
  DMatrix<double> m(3, 3);
  m <<
    3.0, 1.0, 1.0,
    1.0, 2.0, 2.0,
    1.0, 2.0, 2.0;

  auto [actual_values, actual_vectors] = eigen(m);

  DVector<double> expected_values(3);
  expected_values <<
    0.0, 2.0, 5.0;

  DMatrix<double> expected_vectors(3, 3);
  expected_vectors <<
    00.000000, -0.816497, 0.57735,
    -0.707107,  0.408248, 0.57735,
    00.707107,  0.408248, 0.57735;

  ASSERT_EQ(expected_values.size(), actual_values.size());
  ASSERT_EQ(expected_values.rows(), actual_values.rows());
  ASSERT_EQ(expected_values.cols(), actual_values.cols());
  ASSERT_APPROX(expected_values, actual_values);

  ASSERT_EQ(expected_vectors.size(), actual_vectors.size());
  ASSERT_EQ(expected_vectors.rows(), actual_vectors.rows());
  ASSERT_EQ(expected_vectors.cols(), actual_vectors.cols());
  ASSERT_COLLINEAR(expected_vectors, actual_vectors);

  Eigen::DiagonalMatrix<double, 3> DL;
  DL.diagonal() = actual_values;
  DMatrix<double> Mv = m * actual_vectors;
  DMatrix<double> Lv =  actual_vectors * DL;
  ASSERT_APPROX(Mv, Lv);
}

TEST(DMatrix, EigenAsymmetricRealNixedEigenvalues) {
  DMatrix<double> m(3, 3);
  m <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  auto [actual_values, actual_vectors] = eigen(m);

  DVector<double> expected_values(3);
  expected_values <<
    0.0, -1.11684, 16.11684;

  DMatrix<double> expected_vectors(3, 3);
  expected_vectors <<
    00.408248,  0.7858302, 0.2319707,
    -0.816497,  0.0867513, 0.5253221,
    00.408248, -0.6123276, 0.8186735;

  ASSERT_EQ(expected_values.size(), actual_values.size());
  ASSERT_EQ(expected_values.rows(), actual_values.rows());
  ASSERT_EQ(expected_values.cols(), actual_values.cols());
  ASSERT_APPROX(expected_values, actual_values);

  ASSERT_EQ(expected_vectors.size(), actual_vectors.size());
  ASSERT_EQ(expected_vectors.rows(), actual_vectors.rows());
  ASSERT_EQ(expected_vectors.cols(), actual_vectors.cols());
  ASSERT_COLLINEAR(expected_vectors, actual_vectors);

  Eigen::DiagonalMatrix<double, 3> DL;
  DL.diagonal() = actual_values;
  DMatrix<double> Mv = m * actual_vectors;
  DMatrix<double> Lv =  actual_vectors * DL;
  ASSERT_APPROX(Mv, Lv);
}

TEST(DMatrix, CollinearAllColumnsCollinearSameDirection) {
  DMatrix<double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 4.0, 12.0,
    6.0, 12.0, 36.0;

  DMatrix<double> b(3, 3);
  b <<
    2.0, 4.0, 12.0,
    4.0, 8.0, 24.0,
    12.0, 24.0, 72.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(DMatrix, CollinearAllColumnsCollinearOppositeDirection) {
  DMatrix<double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 4.0, 12.0,
    6.0, 12.0, 36.0;

  DMatrix<double> b(3, 3);
  b <<
    -1.0, -2.0, -6.0,
    -2.0, -4.0, -12.0,
    -6.0, -12.0, -36.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(DMatrix, CollinearSomeColumnsCollinearOppositeDirection) {
  DMatrix<double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 4.0, 12.0,
    6.0, 12.0, 36.0;

  DMatrix<double> b(3, 3);
  b <<
    1.0, 2.0, -6.0,
    2.0, 4.0, -12.0,
    6.0, 12.0, -36.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(DMatrix, CollinearAllColumnsNonColinear) {
  DMatrix<double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 4.0, 12.0,
    6.0, 12.0, 36.0;

  DMatrix<double> b(3, 3);
  b <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  ASSERT_FALSE(collinear(a, b));
}

TEST(DMatrix, CollinearSomeColumnsNonCollinear) {
  DMatrix<double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 4.0, 12.0,
    6.0, 12.0, 36.0;

  DMatrix<double> b(3, 3);
  b <<
    1.0, 2.0, 6.0,
    2.0, 4.0, 12.0,
    7.0, 12.0, 36.0;

  ASSERT_FALSE(collinear(a, b));
}
