#include <gtest/gtest.h>

#include "stats.hpp"

using namespace stats;
using namespace Eigen;


TEST(StatsSelectGroup, single_group) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    1;

  Data<double> actual = select_group(data, groups, 1);

  Data<double> expected(3, 3);
  expected <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectGroup, multiple_groups_adjacent) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    2;

  Data<double> actual = select_group(data, groups, 1);

  Data<double> expected(2, 3);
  expected <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectGroup, multiple_groups_mixed) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    2,
    1;

  Data<double> actual = select_group(data, groups, 1);

  Data<double> expected(2, 3);
  expected <<
    1.0, 2.0, 6.0,
    3.0, 4.0, 8.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectGroup, empty_result) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    1;

  Data<double> actual = select_group(data, groups, 2);

  ASSERT_EQ(0, actual.size());
  ASSERT_EQ(0, actual.rows());
  ASSERT_EQ(0, actual.cols());
}

TEST(StatsRemoveGroup, single_group) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    1;

  Data<double> actual = remove_group(data, groups, 1);

  ASSERT_EQ(0, actual.size());
  ASSERT_EQ(0, actual.rows());
  ASSERT_EQ(0, actual.cols());
}

TEST(StatsRemoveGroup, multiple_groups_adjacent1) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    2;

  Data<double> actual = remove_group(data, groups, 1);

  Data<double> expected(1, 3);
  expected <<
    3.0, 4.0, 8.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsRemoveGroup, multiple_groups_adjacent2) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    2;

  Data<double> actual = remove_group(data, groups, 2);

  Data<double> expected(2, 3);
  expected <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsRemoveGroup, multiple_mixed1) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    2,
    1;

  Data<double> actual = remove_group(data, groups, 1);

  Data<double> expected(1, 3);
  expected <<
    2.0, 3.0, 7.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsRemoveGroup, multiple_mixed2) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    2,
    1;

  Data<double> actual = remove_group(data, groups, 2);
  Data<double> expected(2, 3);
  expected <<
    1.0, 2.0, 6.0,
    3.0, 4.0, 8.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsRemoveGroup, non_existent_group) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    2,
    1;

  Data<double> actual = remove_group(data, groups, 3);
  Data<double> expected(3, 3);
  expected <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}


TEST(StatsUnique, empty_result) {
  DataColumn<int> column(0);
  std::set<int> actual = unique(column);
  std::set<int> expected;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(StatsUnique, single_value) {
  DataColumn<int> column(1);
  column << 1;
  std::set<int> actual = unique(column);
  std::set<int> expected = { 1 };

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(StatsUnique, single_value_repeated) {
  DataColumn<int> column(3);
  column <<
    1,
    1,
    1;
  std::set<int> actual = unique(column);
  std::set<int> expected = { 1 };

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(StatsUnique, multiple_values) {
  DataColumn<int> column(3);
  column <<
    1,
    2,
    3;
  std::set<int> actual = unique(column);
  std::set<int> expected = { 1, 2, 3 };

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(StatsUnique, multiple_values_repeated) {
  DataColumn<int> column(3);
  column <<
    1,
    2,
    1;
  std::set<int> actual = unique(column);
  std::set<int> expected = { 1, 2 };

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}


TEST(StatsBetweenGroupsSumOfSquares, single_group) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    0,
    0,
    0;

  Data<double> actual = between_groups_sum_of_squares(data, groups, { 0 });
  Data<double> expected(3, 3);
  expected <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsBetweenGroupsSumOfSquares, two_equal_groups) {
  Data<double> data(6, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0,
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  Data<double> actual = between_groups_sum_of_squares(data, groups, { 0, 1 });
  Data<double> expected(3, 3);
  expected <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;


  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsBetweenGroupsSumOfSquares, multiple_groups_univariate) {
  Data<double> data(8, 1);
  data <<
    23.0,
    25.0,
    18.0,
    29.0,
    19.0,
    21.0,
    35.0,
    17.0;

  DataColumn<int> groups(8);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2;

  Data<double> actual = between_groups_sum_of_squares(data, groups, { 0, 1, 2 });
  Data<double> expected(1, 1);
  expected <<
    19.875;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsBetweenGroupsSumOfSquares, multiple_groups_multivariate) {
  Data<double> data(8, 3);
  data <<
    23.0, 1.0, 1.0,
    25.0, 1.0, 1.0,
    18.0, 1.0, 1.0,
    29.0, 1.0, 1.0,
    19.0, 1.0, 1.0,
    21.0, 1.0, 1.0,
    35.0, 1.0, 1.0,
    17.0, 1.0, 1.0;

  DataColumn<int> groups(8);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2;

  Data<double> actual = between_groups_sum_of_squares(data, groups, { 0, 1, 2 });
  Data<double> expected(3, 3);
  expected <<
    19.875, 0.0, 0.0,
    0.0,    0.0, 0.0,
    0.0,    0.0, 0.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsWithinGroupsSumOfSquares, single_group_no_variance) {
  Data<double> data(3, 3);
  data <<
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0;

  DataColumn<int> groups(3);
  groups <<
    0,
    0,
    0;

  Data<double> actual = within_groups_sum_of_squares(data, groups, { 0 });
  Data<double> expected(3, 3);
  expected <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsWithinGroupsSumOfSquares, single_group_with_variance) {
  Data<double> data(3, 3);
  data <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0;

  DataColumn<int> groups(3);
  groups <<
    0,
    0,
    0;

  Data<double> actual = within_groups_sum_of_squares(data, groups, { 0 });
  Data<double> expected(3, 3);
  expected <<
    2.0, 2.0, 2.0,
    2.0, 2.0, 2.0,
    2.0, 2.0, 2.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsWithinGroupsSumOfSquares, two_equal_groups) {
  Data<double> data(6, 3);
  data <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0;

  DataColumn<int> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  Data<double> actual = within_groups_sum_of_squares(data, groups, { 0, 1 });
  Data<double> expected(3, 3);
  expected <<
    4.0, 4.0, 4.0,
    4.0, 4.0, 4.0,
    4.0, 4.0, 4.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsWithinGroupsSumOfSquares, two_groups_same_variance) {
  Data<double> data(6, 3);
  data <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    4.0, 4.0, 4.0,
    5.0, 5.0, 5.0,
    6.0, 6.0, 6.0;

  DataColumn<int> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  Data<double> actual = within_groups_sum_of_squares(data, groups, { 0, 1 });
  Data<double> expected(3, 3);
  expected <<
    4.0, 4.0, 4.0,
    4.0, 4.0, 4.0,
    4.0, 4.0, 4.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsWithinGroupsSumOfSquares, two_groups_different_variance) {
  Data<double> data(6, 3);
  data <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    1.0, 1.0, 1.0,
    5.0, 5.0, 5.0,
    6.0, 6.0, 6.0;

  DataColumn<int> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  Data<double> actual = within_groups_sum_of_squares(data, groups, { 0, 1 });
  Data<double> expected(3, 3);
  expected <<
    16.0, 16.0, 16.0,
    16.0, 16.0, 16.0,
    16.0, 16.0, 16.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsWithinGroupsSumOfSquares, multiple_groups_multivariate1) {
  Data<double> data(8, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 2.0, 1.0,
    4.0, 3.0, 2.0,
    5.0, 4.0, 3.0,
    9.0, 8.0, 7.0,
    6.0, 5.0, 4.0;

  DataColumn<int> groups(8);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2;

  Data<double> actual = within_groups_sum_of_squares(data, groups, { 0, 1, 2 });
  Data<double> expected(3, 3);
  expected <<
    24.5, 24.5, 24.5,
    24.5, 24.5, 24.5,
    24.5, 24.5, 24.5;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsWithinGroupsSumOfSquares, multiple_groups_multivariate2) {
  Data<double> data(8, 4);
  data <<
    1.0, 2.0, 3.0, 0.0,
    4.0, 5.0, 6.0, 0.0,
    7.0, 8.0, 9.0, 0.0,
    3.0, 2.0, 1.0, 0.0,
    4.0, 3.0, 2.0, 0.0,
    5.0, 4.0, 3.0, 0.0,
    9.0, 8.0, 7.0, 0.0,
    6.0, 5.0, 4.0, 0.0;

  DataColumn<int> groups(8);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2;

  Data<double> actual = within_groups_sum_of_squares(data, groups, { 0, 1, 2 });
  Data<double> expected(4, 4);
  expected <<
    24.5, 24.5, 24.5, 0.0,
    24.5, 24.5, 24.5, 0.0,
    24.5, 24.5, 24.5, 0.0,
    0.0,  0.0,  0.0,  0.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}
