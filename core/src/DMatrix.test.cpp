#include <gtest/gtest.h>

#include "DMatrix.hpp"

using namespace models::math;

#define ASSERT_APPROX(a, b)    ASSERT_TRUE(a.isApprox(b, 0.00001)) << "Expected " << std::endl << a << std::endl << " to be approximate to " << std::endl << b
#define ASSERT_COLLINEAR(a, b) ASSERT_TRUE(collinear(a, b)) << "Expected columns of " << std::endl << a << std::endl << " to be collinear with its respective column of " << std::endl << b

TEST(DMatrix, InnerProductEqualMatrices_unweighted) {
  DMatrix<long double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<long double> b(3, 3);
  b <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<long double> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<long double> actual = inner_product(a, b, weights);
  DMatrix<long double> expected(3, 3);
  expected <<
    14.0, 20.0, 44.0,
    20.0, 29.0, 65.0,
    44.0, 65.0, 149.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, InnerProductEqualMatrices_unweighted_implicit) {
  DMatrix<long double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<long double> b(3, 3);
  b <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;


  DMatrix<long double> actual = inner_product(a, b);
  DMatrix<long double> expected(3, 3);
  expected <<
    14.0, 20.0, 44.0,
    20.0, 29.0, 65.0,
    44.0, 65.0, 149.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, InnerProductEqualMatrices_weighted) {
  DMatrix<long double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<long double> b(3, 3);
  b <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<long double> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  DMatrix<long double> actual = inner_product(a, b, weights);
  DMatrix<long double> expected(3, 3);
  expected <<
    529.0,  736.0,  1564.0,
    736.0,  1024.0, 2176.0,
    1564.0, 2176.0, 4624.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, InnerProductZeroMatrices_unweighted) {
  DMatrix<long double> a(3, 3);
  a <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<long double> b(3, 3);
  b <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<long double> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<long double> actual = inner_product(a, b, weights);
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

TEST(DMatrix, InnerProductZeroMatrices_unweighted_implicit) {
  DMatrix<long double> a(3, 3);
  a <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<long double> b(3, 3);
  b <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<long double> actual = inner_product(a, b);
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

TEST(DMatrix, InnerProductZeroMatrices_weighted) {
  DMatrix<long double> a(3, 3);
  a <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<long double> b(3, 3);
  b <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<long double> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  DMatrix<long double> actual = inner_product(a, b, weights);
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

TEST(DMatrix, InnerProductDifferentMatrices_unweighted) {
  DMatrix<long double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<long double> b(3, 3);
  b <<
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0,
    4.0, 5.0, 9.0;

  DMatrix<long double> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<long double> actual = inner_product(a, b, weights);
  DMatrix<long double> expected(3, 3);
  expected <<
    20.0, 26.0, 50.0,
    29.0, 38.0, 74.0,
    65.0, 86.0, 170.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, InnerProductDifferentMatrices_unweighted_implicit) {
  DMatrix<long double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<long double> b(3, 3);
  b <<
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0,
    4.0, 5.0, 9.0;

  DMatrix<long double> actual = inner_product(a, b);
  DMatrix<long double> expected(3, 3);
  expected <<
    20.0, 26.0, 50.0,
    29.0, 38.0, 74.0,
    65.0, 86.0, 170.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, InnerProductDifferentMatrices_weighted) {
  DMatrix<long double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<long double> b(3, 3);
  b <<
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0,
    4.0, 5.0, 9.0;

  DMatrix<long double> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  DMatrix<long double> actual = inner_product(a, b, weights);
  DMatrix<long double> expected(3, 3);
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
  DMatrix<long double> m(3, 3);
  m <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<long double> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<long double> actual = inner_square(m, weights);
  DMatrix<long double> expected(3, 3);
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
  DMatrix<long double> m(3, 3);
  m <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<long double> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  DMatrix<long double> actual = inner_square(m, weights);
  DMatrix<long double> expected(3, 3);
  expected <<
    529.0,  736.0,  1564.0,
    736.0,  1024.0, 2176.0,
    1564.0, 2176.0, 4624.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, DeterminantGenericPositive_determinant) {
  DMatrix<long double> m(3, 3);
  m <<
    6.0, 1.0, 4.0,
    4.0, 8.0, 4.0,
    6.0, 3.0, 5.0;

  long double actual = determinant(m);
  long double expected = 28.0;

  ASSERT_DOUBLE_EQ(expected, actual);
}

TEST(DMatrix, DeterminantGenericNegative_determinant) {
  DMatrix<long double> m(3, 3);
  m <<
    6.0, 1.0, 4.0,
    4.0, 8.0, 4.0,
    8.0, 3.0, 5.0;

  long double actual = determinant(m);
  long double expected = -28.0;

  ASSERT_DOUBLE_EQ(expected, actual);
}

TEST(DMatrix, DeterminantZeroMatrix) {
  DMatrix<long double> m(3, 3);
  m <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  long double actual = determinant(m);
  long double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, DeterminantSingularMatrix) {
  DMatrix<long double> m(3, 3);
  m <<
    1.0, 2.0, 6.0,
    2.0, 4.0, 7.0,
    3.0, 6.0, 8.0;

  long double actual = determinant(m);
  long double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(DMatrix, SolveIdentityIdentity) {
  DMatrix<long double> l(3, 3);
  l <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<long double> r(3, 3);
  r <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<long double> actual = solve(l, r);
  DMatrix<long double> expected(3, 3);
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
  DMatrix<long double> l(3, 3);
  l <<
    0.0, 1.0, 2.0,
    1.0, 2.0, 3.0,
    3.0, 1.0, 1.0;

  DMatrix<long double> r(3, 3);
  r <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<long double> actual = solve(l, r);
  DMatrix<long double> expected(3, 3);
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
  DMatrix<long double> m(3, 3);
  m <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  auto [actual_values, actual_vectors] = eigen(m);

  DVector<long double> expected_values(3);
  expected_values <<
    1.0, 1.0, 1.0;

  DMatrix<long double> expected_vectors(3, 3);
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
  DMatrix<long double> m(3, 3);
  m <<
    1.0, 0.0, 0.0,
    0.0, 2.0, 0.0,
    0.0, 0.0, 3.0;

  auto [actual_values, actual_vectors] = eigen(m);

  DVector<long double> expected_values(3);
  expected_values <<
    1.0, 2.0, 3.0;

  DMatrix<long double> expected_vectors(3, 3);
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
  DMatrix<long double> m(3, 3);
  m <<
    3.0, 1.0, 1.0,
    1.0, 2.0, 2.0,
    1.0, 2.0, 2.0;

  auto [actual_values, actual_vectors] = eigen(m);

  DVector<long double> expected_values(3);
  expected_values <<
    0.0, 2.0, 5.0;

  DMatrix<long double> expected_vectors(3, 3);
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

  Eigen::DiagonalMatrix<long double, 3> DL;
  DL.diagonal() = actual_values;
  DMatrix<long double> Mv = m * actual_vectors;
  DMatrix<long double> Lv =  actual_vectors * DL;
  ASSERT_APPROX(Mv, Lv);
}

TEST(DMatrix, EigenAsymmetricRealNixedEigenvalues) {
  DMatrix<long double> m(3, 3);
  m <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  auto [actual_values, actual_vectors] = eigen(m);

  DVector<long double> expected_values(3);
  expected_values <<
    0.0, -1.11684, 16.11684;

  DMatrix<long double> expected_vectors(3, 3);
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

  Eigen::DiagonalMatrix<long double, 3> DL;
  DL.diagonal() = actual_values;
  DMatrix<long double> Mv = m * actual_vectors;
  DMatrix<long double> Lv =  actual_vectors * DL;
  ASSERT_APPROX(Mv, Lv);
}

TEST(DMatrix, CollinearAllColumnsCollinearSameDirection) {
  DMatrix<long double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 4.0, 12.0,
    6.0, 12.0, 36.0;

  DMatrix<long double> b(3, 3);
  b <<
    2.0, 4.0, 12.0,
    4.0, 8.0, 24.0,
    12.0, 24.0, 72.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(DMatrix, CollinearAllColumnsCollinearOppositeDirection) {
  DMatrix<long double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 4.0, 12.0,
    6.0, 12.0, 36.0;

  DMatrix<long double> b(3, 3);
  b <<
    -1.0, -2.0, -6.0,
    -2.0, -4.0, -12.0,
    -6.0, -12.0, -36.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(DMatrix, CollinearSomeColumnsCollinearOppositeDirection) {
  DMatrix<long double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 4.0, 12.0,
    6.0, 12.0, 36.0;

  DMatrix<long double> b(3, 3);
  b <<
    1.0, 2.0, -6.0,
    2.0, 4.0, -12.0,
    6.0, 12.0, -36.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(DMatrix, CollinearAllColumnsNonColinear) {
  DMatrix<long double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 4.0, 12.0,
    6.0, 12.0, 36.0;

  DMatrix<long double> b(3, 3);
  b <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  ASSERT_FALSE(collinear(a, b));
}

TEST(DMatrix, CollinearSomeColumnsNonCollinear) {
  DMatrix<long double> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 4.0, 12.0,
    6.0, 12.0, 36.0;

  DMatrix<long double> b(3, 3);
  b <<
    1.0, 2.0, 6.0,
    2.0, 4.0, 12.0,
    7.0, 12.0, 36.0;

  ASSERT_FALSE(collinear(a, b));
}

TEST(DMatrix, SumZero) {
  DMatrix<long double> m(3, 3);
  m <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  long double actual = sum(m);

  ASSERT_EQ(0, actual);
}

TEST(DMatrix, SumIdentity) {
  DMatrix<long double> m(3, 3);
  m <<
    1, 0, 0,
    0, 1, 0,
    0, 0, 1;

  long double actual = sum(m);

  ASSERT_EQ(3, actual);
}

TEST(DMatrix, SumOnes) {
  DMatrix<long double> m(3, 3);
  m <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  long double actual = sum(m);

  ASSERT_EQ(9, actual);
}

TEST(DMatrix, SumGeneric) {
  DMatrix<long double> m(3, 3);
  m <<
    1, 2, 3,
    4, 5, 6,
    7, 8, 9;

  long double actual = sum(m);

  ASSERT_EQ(45, actual);
}

TEST(DMatrix, TraceZero) {
  DMatrix<long double> m(3, 3);
  m <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  long double actual = trace(m);

  ASSERT_EQ(0, actual);
}

TEST(DMatrix, TraceIdentity) {
  DMatrix<long double> m(3, 3);
  m <<
    1, 0, 0,
    0, 1, 0,
    0, 0, 1;

  long double actual = trace(m);

  ASSERT_EQ(3, actual);
}

TEST(DMatrix, TraceZeroDiagonal) {
  DMatrix<long double> m(3, 3);
  m <<
    0, 1, 1,
    1, 0, 1,
    1, 1, 0;

  long double actual = trace(m);

  ASSERT_EQ(0, actual);
}

TEST(DMatrix, TraceGeneric) {
  DMatrix<long double> m(3, 3);
  m <<
    1, 2, 3,
    4, 5, 6,
    7, 8, 9;

  long double actual = trace(m);

  ASSERT_EQ(15, actual);
}
