#include <gtest/gtest.h>

#include "stat.hpp"

TEST(StatMean, single_observation) {
  DMatrix<double> data(1, 3);
  data <<
    1.0, 2.0, 6.0;

  DVector<double> actual = mean(data);

  DVector<double> expected(3);
  expected <<
    1.0, 2.0, 6.0;

  ASSERT_EQ(expected, actual);
}

TEST(StatMean, multiple_equal_observations) {
  DMatrix<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0;

  DVector<double> actual = mean(data);

  DVector<double> expected(3);
  expected <<
    1.0, 2.0, 6.0;

  ASSERT_EQ(expected, actual);
}

TEST(StatMean, multiple_different_observations) {
  DMatrix<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DVector<double> actual = mean(data);

  DVector<double> expected(3);
  expected <<
    2.0, 3.0, 7.0;

  ASSERT_EQ(expected, actual);
}