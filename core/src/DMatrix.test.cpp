#include <gtest/gtest.h>

#include "DMatrix.hpp"

#include "Macros.hpp"

using namespace models::math;

TEST(DMatrix, InnerProductEqualMatricesUnweighted) {
  DMatrix<float> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<float> b(3, 3);
  b <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<float> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<float> actual = inner_product(a, b, weights);
  DMatrix<float> expected(3, 3);
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
  DMatrix<float> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<float> b(3, 3);
  b <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;


  DMatrix<float> actual = inner_product(a, b);
  DMatrix<float> expected(3, 3);
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
  DMatrix<float> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<float> b(3, 3);
  b <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<float> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  DMatrix<float> actual = inner_product(a, b, weights);
  DMatrix<float> expected(3, 3);
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
  DMatrix<float> a(3, 3);
  a <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<float> b(3, 3);
  b <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<float> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<float> actual = inner_product(a, b, weights);
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

TEST(DMatrix, InnerProductZeroMatricesUnweightedImplicit) {
  DMatrix<float> a(3, 3);
  a <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<float> b(3, 3);
  b <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<float> actual = inner_product(a, b);
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

TEST(DMatrix, InnerProductZeroMatricesWeighted) {
  DMatrix<float> a(3, 3);
  a <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<float> b(3, 3);
  b <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  DMatrix<float> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  DMatrix<float> actual = inner_product(a, b, weights);
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

TEST(DMatrix, InnerProductDifferentMatricesUnweighted) {
  DMatrix<float> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<float> b(3, 3);
  b <<
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0,
    4.0, 5.0, 9.0;

  DMatrix<float> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<float> actual = inner_product(a, b, weights);
  DMatrix<float> expected(3, 3);
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
  DMatrix<float> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<float> b(3, 3);
  b <<
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0,
    4.0, 5.0, 9.0;

  DMatrix<float> actual = inner_product(a, b);
  DMatrix<float> expected(3, 3);
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
  DMatrix<float> a(3, 3);
  a <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<float> b(3, 3);
  b <<
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0,
    4.0, 5.0, 9.0;

  DMatrix<float> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  DMatrix<float> actual = inner_product(a, b, weights);
  DMatrix<float> expected(3, 3);
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
  DMatrix<float> m(3, 3);
  m <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<float> weights(3, 3);
  weights <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0;

  DMatrix<float> actual = inner_square(m, weights);
  DMatrix<float> expected(3, 3);
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
  DMatrix<float> m(3, 3);
  m <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<float> actual = inner_square(m);
  DMatrix<float> expected(3, 3);
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
  DMatrix<float> m(3, 3);
  m <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DMatrix<float> weights(3, 3);
  weights <<
    1.0, 2.0,  6.0,
    2.0, 4.0,  12.0,
    6.0, 12.0, 36.0;

  DMatrix<float> actual = inner_square(m, weights);
  DMatrix<float> expected(3, 3);
  expected <<
    529.0,  736.0,  1564.0,
    736.0,  1024.0, 2176.0,
    1564.0, 2176.0, 4624.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}
