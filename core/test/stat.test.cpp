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

  ASSERT_EQ(expected.size(), actual.size());
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

  ASSERT_EQ(expected.size(), actual.size());
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

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(StatSelectGroup, single_group) {
  DMatrix<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DVector<unsigned short> groups(3);
  groups <<
    1,
    1,
    1;

  DMatrix<double> actual = select_group(data, groups, 1);

  DMatrix<double> expected(3, 3);
  expected <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(StatSelectGroup, multiple_groups_adjacent) {
  DMatrix<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DVector<unsigned short> groups(3);
  groups <<
    1,
    1,
    2;

  DMatrix<double> actual = select_group(data, groups, 1);

  DMatrix<double> expected(2, 3);
  expected <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(StatSelectGroup, multiple_groups_mixed) {
  DMatrix<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DVector<unsigned short> groups(3);
  groups <<
    1,
    2,
    1;

  DMatrix<double> actual = select_group(data, groups, 1);

  DMatrix<double> expected(2, 3);
  expected <<
    1.0, 2.0, 6.0,
    3.0, 4.0, 8.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(StatSelectGroup, empty_result) {
  DMatrix<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DVector<unsigned short> groups(3);
  groups <<
    1,
    1,
    1;

  DMatrix<double> actual = select_group(data, groups, 2);

  ASSERT_EQ(0, actual.size());
}

TEST(StatBetweenGroupsSumOfSquares, single_group) {
  DMatrix<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DVector<unsigned short> groups(3);
  groups <<
    0,
    0,
    0;

  double actual = between_groups_sum_of_squares(data, groups, 1);
  double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(StatBetweenGroupsSumOfSquares, two_equal_groups) {
  DMatrix<double> data(6, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0,
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DVector<unsigned short> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  double actual = between_groups_sum_of_squares(data, groups, 2);
  double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(StatBetweenGroupsSumOfSquares, multiple_groups_univariate) {
  DMatrix<double> data(8, 1);
  data <<
    23.0,
    25.0,
    18.0,
    29.0,
    19.0,
    21.0,
    35.0,
    17.0;

  DVector<unsigned short> groups(8);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2;

  double actual = between_groups_sum_of_squares(data, groups, 3);
  double expected = 19.875;

  ASSERT_EQ(expected, actual);
}

TEST(StatBetweenGroupsSumOfSquares, multiple_groups_multivariate) {
  DMatrix<double> data(8, 3);
  data <<
    23.0, 1.0, 1.0,
    25.0, 1.0, 1.0,
    18.0, 1.0, 1.0,
    29.0, 1.0, 1.0,
    19.0, 1.0, 1.0,
    21.0, 1.0, 1.0,
    35.0, 1.0, 1.0,
    17.0, 1.0, 1.0;

  DVector<unsigned short> groups(8);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2;

  double actual = between_groups_sum_of_squares(data, groups, 3);
  double expected = 19.875;

  ASSERT_EQ(expected, actual);
}

TEST(StatWithinGroupsSumOfSquares, single_group_no_variance) {
  DMatrix<double> data(3, 3);
  data <<
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0;

  DVector<unsigned short> groups(3);
  groups <<
    0,
    0,
    0;

  double actual = within_groups_sum_of_squares(data, groups, 1);
  double expected = 0.0;

  ASSERT_EQ(expected, actual);
}

TEST(StatWithinGroupsSumOfSquares, single_group_with_variance) {
  DMatrix<double> data(3, 3);
  data <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0;

  DVector<unsigned short> groups(3);
  groups <<
    0,
    0,
    0;

  double actual = within_groups_sum_of_squares(data, groups, 1);
  double expected = 6.0;

  ASSERT_EQ(expected, actual);
}

TEST(StatWithinGroupsSumOfSquares, two_equal_groups) {
  DMatrix<double> data(6, 3);
  data <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0;

  DVector<unsigned short> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  double actual = within_groups_sum_of_squares(data, groups, 2);
  double expected = 12.0;

  ASSERT_EQ(expected, actual);
}

TEST(StatWithinGroupsSumOfSquares, two_groups_same_variance) {
  DMatrix<double> data(6, 3);
  data <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    4.0, 4.0, 4.0,
    5.0, 5.0, 5.0,
    6.0, 6.0, 6.0;

  DVector<unsigned short> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  double actual = within_groups_sum_of_squares(data, groups, 2);
  double expected = 12.0;

  ASSERT_EQ(expected, actual);
}

TEST(StatWithinGroupsSumOfSquares, two_groups_different_variance) {
  DMatrix<double> data(6, 3);
  data <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    1.0, 1.0, 1.0,
    5.0, 5.0, 5.0,
    6.0, 6.0, 6.0;

  DVector<unsigned short> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  double actual = within_groups_sum_of_squares(data, groups, 2);
  double expected = 6.0 + 42.0;

  ASSERT_EQ(expected, actual);
}

TEST(StatWithinGroupsSumOfSquares, multiple_groups_multivariate) {
  DMatrix<double> data(8, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 2.0, 1.0,
    4.0, 3.0, 2.0,
    5.0, 4.0, 3.0,
    9.0, 8.0, 7.0,
    6.0, 5.0, 4.0;

  DVector<unsigned short> groups(8);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2;

  double actual = within_groups_sum_of_squares(data, groups, 3);
  double expected = 54.0 + 6.0 + 13.5;

  ASSERT_EQ(expected, actual);
}
