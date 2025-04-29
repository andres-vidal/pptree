#include <gtest/gtest.h>

#include "Data.hpp"

#include "Macros.hpp"

using namespace models::stats;

TEST(Data, ShuffleColumnOfDataFirstColumn) {
  Random::seed(0);

  Data<float> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  Data<float> shuffled = shuffle_column(data, 0);

  Data<float> expected(3, 3);
  expected <<
    2.0, 2.0, 6.0,
    1.0, 3.0, 7.0,
    3.0, 4.0, 8.0;
  ASSERT_EQ(expected.size(), shuffled.size());
  ASSERT_EQ(expected.rows(), shuffled.rows());
  ASSERT_EQ(expected.cols(), shuffled.cols());
  ASSERT_EQ(expected, shuffled);
}

TEST(Data, ShuffleColumnOfDataMiddleColumn) {
  Random::seed(0);

  Data<float> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  Data<float> shuffled = shuffle_column(data, 1);

  Data<float> expected(3, 3);
  expected <<
    1.0, 3.0, 6.0,
    2.0, 2.0, 7.0,
    3.0, 4.0, 8.0;

  ASSERT_EQ(expected.size(), shuffled.size());
  ASSERT_EQ(expected.rows(), shuffled.rows());
  ASSERT_EQ(expected.cols(), shuffled.cols());
  ASSERT_EQ(expected, shuffled);
}

TEST(Data, ShuffleColumnOfDataLastColumn) {
  Random::seed(0);

  Data<float> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  Data<float> shuffled = shuffle_column(data, 2);

  Data<float> expected(3, 3);
  expected <<
    1.0, 2.0, 7.0,
    2.0, 3.0, 6.0,
    3.0, 4.0, 8.0;

  ASSERT_EQ(expected.size(), shuffled.size());
  ASSERT_EQ(expected.rows(), shuffled.rows());
  ASSERT_EQ(expected.cols(), shuffled.cols());
  ASSERT_EQ(expected, shuffled);
}


TEST(Data, Standardize) {
  Data<float> data(3, 3);
  data <<
    1.0, 3.0, 1.0,
    2.0, 2.0, 3.0,
    3.0, 1.0, 2.0;

  Data<float> standardized = standardize(data);

  Data<float> expected(3, 3);
  expected <<
    -1.0, 1.0, -1.0,
    0.0, 0.0,  1.0,
    1.0, -1.0, 0.0;

  ASSERT_EQ(expected.size(), standardized.size());
  ASSERT_EQ(expected.rows(), standardized.rows());
  ASSERT_EQ(expected.cols(), standardized.cols());
  ASSERT_EQ(expected, standardized);
}

TEST(Data, Sort) {
  Data<float> x(3, 3);
  x <<
    1.0, 3.0, 1.0,
    2.0, 2.0, 3.0,
    3.0, 1.0, 2.0;

  DataColumn<int> y(3);
  y <<
    1.0,
    2.0,
    1.0;

  sort(x, y);

  Data<float> expected_x(3, 3);
  expected_x <<
    1.0, 3.0, 1.0,
    3.0, 1.0, 2.0,
    2.0, 2.0, 3.0;

  DataColumn<int> expected_y(3);
  expected_y <<
    1.0,
    1.0,
    2.0;

  ASSERT_EQ_MATRIX(expected_x, x);
  ASSERT_EQ_MATRIX(expected_y, y);
}
