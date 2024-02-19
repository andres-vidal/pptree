#include <gtest/gtest.h>

#include "linalg.hpp"

using namespace linalg;
using namespace Eigen;

#define ASSERT_APPROX(a, b) ASSERT_TRUE(a.isApprox(b, 0.00001)) << "Expected " << std::endl << a << std::endl << " to be approximate to " << std::endl << b


TEST(LinAlgMean, single_observation) {
  DMatrix<long double> data(1, 3);
  data <<
    1.0, 2.0, 6.0;

  DVector<long double> actual = mean(data);

  DVector<long double> expected(3);
  expected <<
    1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(LinAlgMean, multiple_equal_observations) {
  DMatrix<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0;

  DVector<long double> actual = mean(data);

  DVector<long double> expected(3);
  expected <<
    1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(LinAlgMean, multiple_different_observations) {
  DMatrix<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DVector<long double> actual = mean(data);

  DVector<long double> expected(3);
  expected <<
    2.0, 3.0, 7.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(LinAlgOuterProduct, equal_vectors) {
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

TEST(LinAlgOuterProduct, different_vectors1) {
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

TEST(LinAlgOuterProduct, different_vectors2) {
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

TEST(LinAlgOuterProduct, zero_vectors) {
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

TEST(LinAlgOuterProduct, scalar_vectors) {
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

TEST(LinAlgOuterSquare, generic) {
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

TEST(LinAlgOuterSquare, generic2) {
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

TEST(LinAlgOuterSquare, zero_vector) {
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

TEST(LinAlgInnerProduct, equal_vectors_unweighted) {
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

TEST(LinAlgInnerProduct, equal_vectors_unweighted_implicit) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> b(3);
  b << 1.0, 2.0, 6.0;

  long double actual = inner_product(a, b);
  long double expected = 41;

  ASSERT_EQ(expected, actual);
}

TEST(LinAlgInnerProduct, equal_vectors_weighted) {
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

TEST(LinAlgInnerProduct, zero_vectors_unweighted) {
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

TEST(LinAlgInnerProduct, zero_vectors_unweighted_implicit) {
  DVector<long double> a(3);
  a << 0.0, 0.0, 0.0;

  DVector<long double> b(3);
  b << 0.0, 0.0, 0.0;

  long double actual = inner_product(a, b);
  long double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(LinAlgInnerProduct, zero_vectors_weighted) {
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

TEST(LinAlgInnerProduct, different_vectors_unweighted) {
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

TEST(LinAlgInnerProduct, different_vectors_unweighted_implicit) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> b(3);
  b << 2.0, 3.0, 7.0;

  long double actual = inner_product(a, b);
  long double expected = 50;

  ASSERT_EQ(expected, actual);
}

TEST(LinAlgInnerProduct, different_vectors_weighted) {
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

TEST(LinAlgInnerProduct, equal_matrices_unweighted) {
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

TEST(LinAlgInnerProduct, equal_matrices_unweighted_implicit) {
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

TEST(LinAlgInnerProduct, equal_matrices_weighted) {
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

TEST(LinAlgInnerProduct, zero_matrices_unweighted) {
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

TEST(LinAlgInnerProduct, zero_matrices_unweighted_implicit) {
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

TEST(LinAlgInnerProduct, zero_matrices_weighted) {
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

TEST(LinAlgInnerProduct, different_matrices_unweighted) {
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

TEST(LinAlgInnerProduct, different_matrices_unweighted_implicit) {
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

TEST(LinAlgInnerProduct, different_matrices_weighted) {
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

TEST(LinAlgInnerSquare, generic_unweighted) {
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

TEST(LinAlgInnerSquare, generic_weighted) {
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

TEST(LinAlgInnerSquare, generic_matrix_unweighted) {
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

TEST(LinAlgInnerSquare, generic_matrix_weighted) {
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

TEST(LinAlgDeterminant, generic_positive_determinant) {
  DMatrix<long double> m(3, 3);
  m <<
    6.0, 1.0, 4.0,
    4.0, 8.0, 4.0,
    6.0, 3.0, 5.0;

  long double actual = determinant(m);
  long double expected = 28.0;

  ASSERT_DOUBLE_EQ(expected, actual);
}

TEST(LinAlgDeterminant, generic_negative_determinant) {
  DMatrix<long double> m(3, 3);
  m <<
    6.0, 1.0, 4.0,
    4.0, 8.0, 4.0,
    8.0, 3.0, 5.0;

  long double actual = determinant(m);
  long double expected = -28.0;

  ASSERT_DOUBLE_EQ(expected, actual);
}

TEST(LinAlgDeterminant, zero_matrix) {
  DMatrix<long double> m(3, 3);
  m <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  long double actual = determinant(m);
  long double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(LinAlgDeterminant, singular_matrix) {
  DMatrix<long double> m(3, 3);
  m <<
    1.0, 2.0, 6.0,
    2.0, 4.0, 7.0,
    3.0, 6.0, 8.0;

  long double actual = determinant(m);
  long double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(LinAlgInverse, zero_matrix) {
  DMatrix<long double> m(3, 3);
  m <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  ASSERT_DEATH({ inverse(m); }, "Given matrix is not invertible");
}

TEST(LinAlgInverse, singular_matrix) {
  DMatrix<long double> m(3, 3);
  m <<
    1.0, 1.0, 6.0,
    2.0, 2.0, 7.0,
    3.0, 3.0, 8.0;

  ASSERT_DEATH({ inverse(m); }, "Given matrix is not invertible");
}

TEST(LinAlgInverse, identity) {
  DMatrix<long double> m(3, 3);
  m <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<long double> actual = inverse(m);
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

TEST(LinAlgInverse, generic) {
  DMatrix<long double> m(3, 3);
  m <<
    0.0, 1.0, 2.0,
    1.0, 2.0, 3.0,
    3.0, 1.0, 1.0;

  DMatrix<long double> actual = inverse(m);
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

TEST(LinAlgEigen, identity) {
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

TEST(LinAlgEigen, diagonal) {
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

TEST(LinAlgEigen, symmetric_real_non_negative_eigenvalues) {
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
  ASSERT_APPROX(expected_vectors, actual_vectors);

  DiagonalMatrix<long double, 3> DL;
  DL.diagonal() = actual_values;
  DMatrix<long double> Mv = m * actual_vectors;
  DMatrix<long double> Lv =  actual_vectors * DL;
  ASSERT_APPROX(Mv, Lv);
}

TEST(LinAlgEigen, asymmetric_real_mixed_eigenvalues) {
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
  ASSERT_APPROX(expected_vectors, actual_vectors);

  DiagonalMatrix<long double, 3> DL;
  DL.diagonal() = actual_values;
  DMatrix<long double> Mv = m * actual_vectors;
  DMatrix<long double> Lv =  actual_vectors * DL;
  ASSERT_APPROX(Mv, Lv);
}

TEST(LinAlgCollinear, collinear_true_same_direction) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> b(3);
  b << 2.0, 4.0, 12.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(LinalCollinear, collinear_false_opposite_direction) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> b(3);
  b << -1.0, -2.0, -6.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(LinAlgCollinear, collinear_false) {
  DVector<long double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<long double> b(3);
  b << 2.0, 3.0, 7.0;

  ASSERT_FALSE(collinear(a, b));
}
