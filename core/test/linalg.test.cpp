#include <gtest/gtest.h>

#include "linalg.hpp"

using namespace linalg;
using namespace Eigen;


TEST(LinAlgMean, single_observation) {
  DMatrix<double> data(1, 3);
  data <<
    1.0, 2.0, 6.0;

  DVector<double> actual = mean(data);

  DVector<double> expected(3);
  expected <<
    1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(LinAlgMean, multiple_equal_observations) {
  DMatrix<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0;

  DVector<double> actual = mean(data);

  DVector<double> expected(3);
  expected <<
    1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(LinAlgMean, multiple_different_observations) {
  DMatrix<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DVector<double> actual = mean(data);

  DVector<double> expected(3);
  expected <<
    2.0, 3.0, 7.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(LinAlgOuterProduct, equal_vectors) {
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

TEST(LinAlgOuterProduct, different_vectors1) {
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

TEST(LinAlgOuterProduct, different_vectors2) {
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

TEST(LinAlgOuterProduct, zero_vectors) {
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

TEST(LinAlgOuterProduct, scalar_vectors) {
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

TEST(LinAlgOuterSquare, generic) {
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

TEST(LinAlgOuterSquare, generic2) {
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

TEST(LinAlgOuterSquare, zero_vector) {
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

TEST(LinAlgInnerProduct, equal_vectors_unweighted) {
  DVector<double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<double> b(3);
  b << 1.0, 2.0, 6.0;

  DMatrix<double> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  double actual = inner_product(a, b, weights);
  double expected = 41;

  ASSERT_EQ(expected, actual);
}

TEST(LinAlgInnerProduct, equal_vectors_weighted) {
  DVector<double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<double> b(3);
  b << 1.0, 2.0, 6.0;

  DMatrix<double> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  double actual = inner_product(a, b, weights);
  double expected = 1681;

  ASSERT_EQ(expected, actual);
}

TEST(LinAlgInnerProduct, zero_vectors_unweighted) {
  DVector<double> a(3);
  a << 0.0, 0.0, 0.0;

  DVector<double> b(3);
  b << 0.0, 0.0, 0.0;

  DMatrix<double> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  double actual = inner_product(a, b, weights);
  double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(LinAlgInnerProduct, zero_vectors_weighted) {
  DVector<double> a(3);
  a << 0.0, 0.0, 0.0;

  DVector<double> b(3);
  b << 0.0, 0.0, 0.0;

  DMatrix<double> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  double actual = inner_product(a, b, weights);
  double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(LinAlgInnerProduct, different_vectors_unweighted) {
  DVector<double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<double> b(3);
  b << 2.0, 3.0, 7.0;

  DMatrix<double> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  double actual = inner_product(a, b, weights);
  double expected = 50;

  ASSERT_EQ(expected, actual);
}

TEST(LinAlgInnerProduct, different_vectors_weighted) {
  DVector<double> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<double> b(3);
  b << 2.0, 3.0, 7.0;

  DMatrix<double> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  double actual = inner_product(a, b, weights);
  double expected = 2050;

  ASSERT_EQ(expected, actual);
}

TEST(LinAlgInnerProduct, equal_matrices_unweighted) {
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

TEST(LinAlgInnerProduct, equal_matrices_weighted) {
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

TEST(LinAlgInnerProduct, zero_matrices_unweighted) {
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

TEST(LinAlgInnerProduct, zero_matrices_weighted) {
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

TEST(LinAlgInnerProduct, different_matrices_unweighted) {
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

TEST(LinAlgInnerProduct, different_matrices_weighted) {
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

TEST(LinAlgInnerSquare, generic_unweighted) {
  DVector<double> m(3);
  m << 1.0, 2.0, 6.0;

  DMatrix<double> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  double actual = inner_square(m, weights);
  double expected = 41;

  ASSERT_EQ(expected, actual);
}

TEST(LinAlgInnerSquare, generic_weighted) {
  DVector<double> m(3);
  m << 1.0, 2.0, 6.0;

  DMatrix<double> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  double actual = inner_square(m, weights);
  double expected = 1681;

  ASSERT_EQ(expected, actual);
}

TEST(LinAlgInnerSquare, generic_matrix_unweighted) {
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

TEST(LinAlgInnerSquare, generic_matrix_weighted) {
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

TEST(LinAlgDeterminant, generic_positive_determinant) {
  DMatrix<double> m(3, 3);
  m <<
    6.0, 1.0, 4.0,
    4.0, 8.0, 4.0,
    6.0, 3.0, 5.0;

  double actual = determinant(m);
  double expected = 28.0;

  ASSERT_DOUBLE_EQ(expected, actual);
}


TEST(LinAlgDeterminant, generic_negative_determinant) {
  DMatrix<double> m(3, 3);
  m <<
    6.0, 1.0, 4.0,
    4.0, 8.0, 4.0,
    8.0, 3.0, 5.0;

  double actual = determinant(m);
  double expected = -28.0;

  ASSERT_DOUBLE_EQ(expected, actual);
}

TEST(LinAlgDeterminant, zero_matrix) {
  DMatrix<double> m(3, 3);
  m <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  double actual = determinant(m);
  double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(LinAlgDeterminant, singular_matrix) {
  DMatrix<double> m(3, 3);
  m <<
    1.0, 2.0, 6.0,
    2.0, 4.0, 7.0,
    3.0, 6.0, 8.0;

  double actual = determinant(m);
  double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(LinAlgInverse, zero_matrix) {
  DMatrix<double> m(3, 3);
  m <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  ASSERT_THROW({ inverse(m); }, std::invalid_argument);
}

TEST(LinAlgInverse, singular_matrix) {
  DMatrix<double> m(3, 3);
  m <<
    1.0, 1.0, 6.0,
    2.0, 2.0, 7.0,
    3.0, 3.0, 8.0;

  ASSERT_THROW({ inverse(m); }, std::invalid_argument);
}

TEST(LinAlgInverse, identity) {
  DMatrix<double> m(3, 3);
  m <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<double> actual = inverse(m);
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

TEST(LinAlgInverse, generic) {
  DMatrix<double> m(3, 3);
  m <<
    0.0, 1.0, 2.0,
    1.0, 2.0, 3.0,
    3.0, 1.0, 1.0;

  DMatrix<double> actual = inverse(m);
  DMatrix<double> expected(3, 3);
  expected <<
    0.5,  -0.5,   0.5,
    -4.0,  3.0,  -1.0,
    2.5,  -1.5,   0.5;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_TRUE(expected.isApprox(actual));
}

TEST(LinAlgEigen, identity) {
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

TEST(LinAlgEigen, diagonal) {
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

TEST(LinAlgEigen, symmetric_real_non_negative_eigenvalues) {
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
  ASSERT_TRUE(expected_values.isApprox(actual_values));

  ASSERT_EQ(expected_vectors.size(), actual_vectors.size());
  ASSERT_EQ(expected_vectors.rows(), actual_vectors.rows());
  ASSERT_EQ(expected_vectors.cols(), actual_vectors.cols());
  ASSERT_TRUE(expected_vectors.isApprox(actual_vectors, 0.0001));

  DiagonalMatrix<double, 3> DL;
  DL.diagonal() = actual_values;
  DMatrix<double> Mv = m * actual_vectors;
  DMatrix<double> Lv =  actual_vectors * DL;
  ASSERT_TRUE(Mv.isApprox(Lv, 0.0001));
}

TEST(LinAlgEigen, asymmetric_real_mixed_eigenvalues) {
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
  ASSERT_TRUE(expected_values.isApprox(actual_values, 0.0005));

  ASSERT_EQ(expected_vectors.size(), actual_vectors.size());
  ASSERT_EQ(expected_vectors.rows(), actual_vectors.rows());
  ASSERT_EQ(expected_vectors.cols(), actual_vectors.cols());
  ASSERT_TRUE(expected_vectors.isApprox(actual_vectors, 0.0001));

  DiagonalMatrix<double, 3> DL;
  DL.diagonal() = actual_values;
  DMatrix<double> Mv = m * actual_vectors;
  DMatrix<double> Lv =  actual_vectors * DL;
  ASSERT_TRUE(Mv.isApprox(Lv, 0.0001));
}
