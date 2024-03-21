#include <gtest/gtest.h>

#include "stats.hpp"

using namespace stats;
using namespace Eigen;


TEST(StatsSelectGroup, single_group) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    1;

  Data<long double> actual = select_group(data, groups, 1);

  Data<long double> expected(3, 3);
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
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    2;

  Data<long double> actual = select_group(data, groups, 1);

  Data<long double> expected(2, 3);
  expected <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectGroup, multiple_groups_mixed) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    2,
    1;

  Data<long double> actual = select_group(data, groups, 1);

  Data<long double> expected(2, 3);
  expected <<
    1.0, 2.0, 6.0,
    3.0, 4.0, 8.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectGroup, empty_result) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    1;

  Data<long double> actual = select_group(data, groups, 2);

  ASSERT_EQ(0, actual.size());
  ASSERT_EQ(0, actual.rows());
  ASSERT_EQ(0, actual.cols());
}

TEST(StatsSelectGroups, single_on_single) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;
  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    1;

  Data<long double> actual = select_groups(data, groups, { 1 });

  Data<long double> expected(3, 3);
  expected <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectGroups, single_on_single_empty) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;
  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    1;

  Data<long double> actual = select_groups(data, groups, { 2 });

  ASSERT_EQ(0, actual.size());
  ASSERT_EQ(0, actual.rows());
  ASSERT_EQ(0, actual.cols());
}

TEST(StatsSelectGroups, single_on_multiple_adjacent) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;
  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    2;

  Data<long double> actual = select_groups(data, groups, { 1 });

  Data<long double> expected(2, 3);
  expected <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectGroups, single_on_multiple_mixed) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;
  DataColumn<int> groups(3);
  groups <<
    1,
    2,
    1;

  Data<long double> actual = select_groups(data, groups, { 1 });

  Data<long double> expected(2, 3);
  expected <<
    1.0, 2.0, 3.0,
    7.0, 8.0, 9.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectGroups, multiple_on_multiple) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;
  DataColumn<int> groups(3);
  groups <<
    1,
    2,
    1;

  Data<long double> actual = select_groups(data, groups, { 1, 2 });

  Data<long double> expected(3, 3);
  expected <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectGroups, multiple_on_multiple_empty) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;
  DataColumn<int> groups(3);
  groups <<
    1,
    2,
    1;

  Data<long double> actual = select_groups(data, groups, { 3, 4 });

  ASSERT_EQ(0, actual.size());
  ASSERT_EQ(0, actual.rows());
  ASSERT_EQ(0, actual.cols());
}

TEST(StatsSelectGroups, multiple_on_single) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;
  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    1;

  Data<long double> actual = select_groups(data, groups, { 1, 2 });

  Data<long double> expected(3, 3);
  expected <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectGroups, multiple_on_single_empty) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;
  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    1;

  Data<long double> actual = select_groups(data, groups, { 3, 4 });

  ASSERT_EQ(0, actual.size());
  ASSERT_EQ(0, actual.rows());
  ASSERT_EQ(0, actual.cols());
}

TEST(StatsSelectGroups, multiple_on_multiple_adjacent) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;
  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    2;

  Data<long double> actual = select_groups(data, groups, { 1, 2 });

  Data<long double> expected(3, 3);
  expected <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectGroup, multiple_on_multiple2) {
  // Use a matrix with 15 rows
  Data<long double> data(15, 3);
  data <<
    1.0,  2.0,  3.0,
    4.0,  5.0,  6.0,
    7.0,  8.0,  9.0,
    10.0, 11.0, 12.0,
    13.0, 14.0, 15.0,
    16.0, 17.0, 18.0,
    19.0, 20.0, 21.0,
    22.0, 23.0, 24.0,
    25.0, 26.0, 27.0,
    28.0, 29.0, 30.0,
    31.0, 32.0, 33.0,
    34.0, 35.0, 36.0,
    37.0, 38.0, 39.0,
    40.0, 41.0, 42.0,
    43.0, 44.0, 45.0;

  DataColumn<int> groups(15);
  groups <<
    1,
    1,
    1,
    2,
    2,
    2,
    3,
    3,
    3,
    4,
    4,
    4,
    5,
    5,
    5;

  Data<long double> actual = select_groups(data, groups, { 1, 2, 4 });

  Data<long double> expected(9, 3);
  expected <<
    1.0,  2.0,  3.0,
    4.0,  5.0,  6.0,
    7.0,  8.0,  9.0,
    10.0, 11.0, 12.0,
    13.0, 14.0, 15.0,
    16.0, 17.0, 18.0,
    28.0, 29.0, 30.0,
    31.0, 32.0, 33.0,
    34.0, 35.0, 36.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsRemoveGroup, single_group) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    1;

  Data<long double> actual = remove_group(data, groups, 1);

  ASSERT_EQ(0, actual.size());
  ASSERT_EQ(0, actual.rows());
  ASSERT_EQ(0, actual.cols());
}

TEST(StatsRemoveGroup, multiple_groups_adjacent1) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    2;

  Data<long double> actual = remove_group(data, groups, 1);

  Data<long double> expected(1, 3);
  expected <<
    3.0, 4.0, 8.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsRemoveGroup, multiple_groups_adjacent2) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    2;

  Data<long double> actual = remove_group(data, groups, 2);

  Data<long double> expected(2, 3);
  expected <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsRemoveGroup, multiple_mixed1) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    2,
    1;

  Data<long double> actual = remove_group(data, groups, 1);

  Data<long double> expected(1, 3);
  expected <<
    2.0, 3.0, 7.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsRemoveGroup, multiple_mixed2) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    2,
    1;

  Data<long double> actual = remove_group(data, groups, 2);
  Data<long double> expected(2, 3);
  expected <<
    1.0, 2.0, 6.0,
    3.0, 4.0, 8.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsRemoveGroup, non_existent_group) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  DataColumn<int> groups(3);
  groups <<
    1,
    2,
    1;

  Data<long double> actual = remove_group(data, groups, 3);
  Data<long double> expected(3, 3);
  expected <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsBinaryRegroup, single_group) {
  Data<long double> data(3, 1);
  data <<
    1.0,
    4.0,
    7.0;
  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    1;

  ASSERT_DEATH({ binary_regroup(data, groups, { 1 }); }, "Must have more than 2 groups to binary regroup");
}

TEST(StatsBinaryRegroup, two_groups) {
  Data<long double> data(3, 1);
  data <<
    1.0,
    4.0,
    7.0;
  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    2;

  ASSERT_DEATH({ binary_regroup(data, groups, { 1, 2 }); }, "Must have more than 2 groups to binary regroup");
}

TEST(StatsBinaryRegroup, multidimensional) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;
  DataColumn<int> groups(3);
  groups <<
    1,
    2,
    3;

  ASSERT_DEATH({ binary_regroup(data, groups, { 1, 2, 3 }); }, "Data must be unidimensional to binary regroup");
}

TEST(StatsBinaryGroup, single_observation_per_group) {
  Data<long double> data(3, 1);
  data <<
    1.0,
    2.0,
    7.0;
  DataColumn<int> groups(3);
  groups <<
    1,
    2,
    3;

  auto [actual_new_groups, actual_new_unique_groups, mapping] = binary_regroup(data, groups, { 1, 2, 3 });

  DataColumn<int> expected_new_groups(3);
  expected_new_groups <<
    0,
    0,
    1;

  std::set<int> expected_new_unique_groups = { 0, 1 };
  std::set<int> expected_0_mapping = { 1, 2 };
  std::set<int> expected_1_mapping = { 3 };

  ASSERT_EQ(expected_0_mapping, mapping[0]);
  ASSERT_EQ(expected_1_mapping, mapping[1]);
  ASSERT_EQ(expected_new_unique_groups, actual_new_unique_groups);
  ASSERT_EQ(expected_new_groups.size(), actual_new_groups.size());
  ASSERT_EQ(expected_new_groups.rows(), actual_new_groups.rows());
  ASSERT_EQ(expected_new_groups.cols(), actual_new_groups.cols());
  ASSERT_EQ(expected_new_groups, actual_new_groups);
}

TEST(StatsBinaryGroup, multiple_observations_per_group_adjacent) {
  Data<long double> data(8, 1);
  data <<
    1.0,
    2.0,
    3.0,
    7.0,
    8.0,
    9.0,
    11.0,
    12.0;

  DataColumn<int> groups(8);
  groups <<
    1,
    1,
    1,
    2,
    2,
    2,
    3,
    3;

  auto [actual_new_groups, actual_new_unique_groups, mapping] = binary_regroup(data, groups, { 1, 2, 3 });

  DataColumn<int> expected_new_groups(8);
  expected_new_groups <<
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1;

  std::set<int> expected_new_unique_groups = { 0, 1 };
  std::set<int> expected_0_mapping = { 1 };
  std::set<int> expected_1_mapping = { 2, 3 };

  ASSERT_EQ(expected_0_mapping, mapping[0]);
  ASSERT_EQ(expected_1_mapping, mapping[1]);
  ASSERT_EQ(expected_new_unique_groups, actual_new_unique_groups);
  ASSERT_EQ(expected_new_groups.size(), actual_new_groups.size());
  ASSERT_EQ(expected_new_groups.rows(), actual_new_groups.rows());
  ASSERT_EQ(expected_new_groups.cols(), actual_new_groups.cols());
  ASSERT_EQ(expected_new_groups, actual_new_groups);
}

TEST(StatsBinaryGroup, multiple_observations_per_group_mixed) {
  Data<long double> data(8, 1);
  data <<
    7.0,
    1.0,
    12.0,
    8.0,
    2.0,
    9.0,
    11.0,
    3.0;

  DataColumn<int> groups(8);
  groups <<
    2,
    1,
    3,
    2,
    1,
    2,
    3,
    1;

  auto [actual_new_groups, actual_new_unique_groups, mapping] = binary_regroup(data, groups, { 1, 2, 3 });

  DataColumn<int> expected_new_groups(8);
  expected_new_groups <<
    1,
    0,
    1,
    1,
    0,
    1,
    1,
    0;

  std::set<int> expected_new_unique_groups = { 0, 1 };
  std::set<int> expected_0_mapping = { 1 };
  std::set<int> expected_1_mapping = { 2, 3 };

  ASSERT_EQ(expected_0_mapping, mapping[0]);
  ASSERT_EQ(expected_1_mapping, mapping[1]);
  ASSERT_EQ(expected_new_unique_groups, actual_new_unique_groups);
  ASSERT_EQ(expected_new_groups.size(), actual_new_groups.size());
  ASSERT_EQ(expected_new_groups.rows(), actual_new_groups.rows());
  ASSERT_EQ(expected_new_groups.cols(), actual_new_groups.cols());
  ASSERT_EQ(expected_new_groups, actual_new_groups);
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
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> groups(3);
  groups <<
    0,
    0,
    0;

  Data<long double> actual = between_groups_sum_of_squares(data, groups, { 0 });
  Data<long double> expected(3, 3);
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
  Data<long double> data(6, 3);
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

  Data<long double> actual = between_groups_sum_of_squares(data, groups, { 0, 1 });
  Data<long double> expected(3, 3);
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
  Data<long double> data(8, 1);
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

  Data<long double> actual = between_groups_sum_of_squares(data, groups, { 0, 1, 2 });
  Data<long double> expected(1, 1);
  expected <<
    19.875;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsBetweenGroupsSumOfSquares, multiple_groups_univariate_non_sequential_group_ids) {
  Data<long double> data(8, 1);
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
    1,
    1,
    1,
    7,
    7,
    7,
    3,
    3;

  Data<long double> actual = between_groups_sum_of_squares(data, groups, { 1, 7, 3 });
  Data<long double> expected(1, 1);
  expected <<
    19.875;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsBetweenGroupsSumOfSquares, multiple_groups_multivariate) {
  Data<long double> data(8, 3);
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

  Data<long double> actual = between_groups_sum_of_squares(data, groups, { 0, 1, 2 });
  Data<long double> expected(3, 3);
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
  Data<long double> data(3, 3);
  data <<
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0;

  DataColumn<int> groups(3);
  groups <<
    0,
    0,
    0;

  Data<long double> actual = within_groups_sum_of_squares(data, groups, { 0 });
  Data<long double> expected(3, 3);
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
  Data<long double> data(3, 3);
  data <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0;

  DataColumn<int> groups(3);
  groups <<
    0,
    0,
    0;

  Data<long double> actual = within_groups_sum_of_squares(data, groups, { 0 });
  Data<long double> expected(3, 3);
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
  Data<long double> data(6, 3);
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

  Data<long double> actual = within_groups_sum_of_squares(data, groups, { 0, 1 });
  Data<long double> expected(3, 3);
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
  Data<long double> data(6, 3);
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

  Data<long double> actual = within_groups_sum_of_squares(data, groups, { 0, 1 });
  Data<long double> expected(3, 3);
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
  Data<long double> data(6, 3);
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

  Data<long double> actual = within_groups_sum_of_squares(data, groups, { 0, 1 });
  Data<long double> expected(3, 3);
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
  Data<long double> data(8, 3);
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

  Data<long double> actual = within_groups_sum_of_squares(data, groups, { 0, 1, 2 });
  Data<long double> expected(3, 3);
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
  Data<long double> data(8, 4);
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

  Data<long double> actual = within_groups_sum_of_squares(data, groups, { 0, 1, 2 });
  Data<long double> expected(4, 4);
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

TEST(StatsSample, negative_sample_size) {
  std::mt19937 generator(0);

  Data<long double> data(0, 0);

  ASSERT_DEATH({ sample(data, -1, generator); }, "Sample size must be greater than 0.");
}

TEST(StatsSample, zero_sample_size) {
  std::mt19937 generator(0);

  Data<long double> data(0, 0);

  ASSERT_DEATH({ sample(data, 0, generator); }, "Sample size must be greater than 0.");
}


TEST(StatsSample, sample_size_larger_than_data_rows) {
  std::mt19937 generator(0);

  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  ASSERT_DEATH({ sample(data, 4, generator); }, "Sample size cannot be larger than the number of rows in the data.");
}

TEST(StatsSample, sample_has_correct_size) {
  std::mt19937 generator(0);

  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  Data<long double> actual = sample(data, 2, generator);

  ASSERT_EQ(2, actual.rows());
  ASSERT_EQ(3, actual.cols());
}

TEST(StatsSample, sample_is_subset_of_data) {
  std::mt19937 generator(0);

  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  Data<long double> actual = sample(data, 2, generator);

  for (int i = 0; i < actual.rows(); i++) {
    bool found = false;

    for (int j = 0; j < data.rows(); j++) {
      if (actual.row(i) == data.row(j)) {
        found = true;
        break;
      }
    }

    ASSERT_TRUE(found) << "Expected to find row [" << actual.row(i) << "] in the original data: " << std::endl << data << std::endl;
  }
}

TEST(StatsStratifiedSample, negative_sample_size) {
  std::mt19937 generator(0);

  Data<long double> data(6, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  std::map<int, int> sizes = { { 0, -1 }, { 1, 2 } };

  ASSERT_DEATH({ stratified_sample(data, groups, sizes, generator); }, "Sample size must be greater than 0.");
}

TEST(StatsStratifiedSample, zero_sample_size) {
  std::mt19937 generator(0);

  Data<long double> data(6, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  std::map<int, int> sizes = { { 0, 0 }, { 1, 2 } };

  ASSERT_DEATH({ stratified_sample(data, groups, sizes, generator); }, "Sample size must be greater than 0.");
}

TEST(StatsStratifiedSample, sample_size_larger_than_group_size) {
  std::mt19937 generator(0);

  Data<long double> data(6, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  std::map<int, int> sizes = { { 0, 4 }, { 1, 2 } };

  ASSERT_DEATH({ stratified_sample(data, groups, sizes, generator); }, "Sample size cannot be larger than the number of rows in the data.");
}

TEST(StatsStratifiedSample, sample_has_correct_size) {
  std::mt19937 generator(0);

  Data<long double> data(6, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  std::map<int, int> sizes = { { 0, 2 }, { 1, 2 } };

  auto [result, result_groups] = stratified_sample(data, groups, sizes, generator);

  ASSERT_EQ(4, result.rows());
  ASSERT_EQ(3, result.cols());
}

TEST(StatsStratifiedSample, sample_has_correct_size_per_strata) {
  std::mt19937 generator(0);

  Data<long double> data(6, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  std::map<int, int> sizes = { { 0, 2 }, { 1, 2 } };

  auto [result, result_groups] = stratified_sample(data, groups, sizes, generator);

  std::map<int, int> result_sizes;

  for (int i = 0; i < result_groups.size(); i++) {
    result_sizes[result_groups[i]]++;
  }

  ASSERT_EQ(2, result_sizes[0]);
  ASSERT_EQ(2, result_sizes[1]);
}

TEST(StatsStratifiedSample, sample_is_subset_of_data_per_strata) {
  std::mt19937 generator(0);

  Data<long double> data(6, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  std::map<int, int> sizes = { { 0, 2 }, { 1, 2 } };

  auto [result, result_groups] = stratified_sample(data, groups, sizes, generator);

  for (int i = 0; i < result.rows(); i++) {
    bool found = false;

    for (int j = 0; j < data.rows(); j++) {
      if (result.row(i) == data.row(j) && result_groups[i] == groups[j]) {
        found = true;
        break;
      }
    }

    ASSERT_TRUE(found) << "Expected to find row [" << result.row(i) << "] in the original data: " << std::endl << data << std::endl;
  }
}

TEST(StatsStratifiedProportionalSample, negative_sample_size) {
  std::mt19937 generator(0);

  Data<long double> data(6, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  ASSERT_DEATH({ stratified_proportional_sample(data, groups, { 0, 1 }, -1, generator); }, "Sample size must be greater than 0.");
}

TEST(StatsStratifiedProportionalSample, zero_sample_size) {
  std::mt19937 generator(0);

  Data<long double> data(6, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  ASSERT_DEATH({ stratified_proportional_sample(data, groups, { 0, 1 }, 0, generator); }, "Sample size must be greater than 0.");
}

TEST(StatsStratifiedProportionalSample, sample_size_larger_than_data_rows) {
  std::mt19937 generator(0);

  Data<long double> data(6, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  ASSERT_DEATH({ stratified_proportional_sample(data, groups, { 0, 1 }, 7, generator); }, "Sample size cannot be larger than the number of rows in the data.");
}

TEST(StatsStratifiedProportionalSample, sample_has_correct_size) {
  std::mt19937 generator(0);

  Data<long double> data(6, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;


  auto [result, result_groups] = stratified_proportional_sample(data, groups, { 0, 1 }, 4, generator);

  ASSERT_EQ(4, result.rows());
  ASSERT_EQ(3, result.cols());
}

TEST(StatsStratifiedProportionalSample, sample_has_correct_size_per_strata) {
  std::mt19937 generator(0);

  Data<long double> data(6, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> groups(6);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1;

  auto [result, result_groups] = stratified_proportional_sample(data, groups, { 0, 1 }, 4, generator);

  std::map<int, int> result_sizes;

  for (int i = 0; i < result_groups.size(); i++) {
    result_sizes[result_groups[i]]++;
  }

  ASSERT_EQ(2, result_sizes[0]);
  ASSERT_EQ(2, result_sizes[1]);
}

TEST(StatsStratifiedProportionalSample, sample_is_subset_of_data_per_strata) {
  std::mt19937 generator(0);

  Data<long double> data(6, 3);
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

  auto [result, result_groups] = stratified_proportional_sample(data, groups, { 0, 1 }, 4, generator);

  for (int i = 0; i < result.rows(); i++) {
    bool found = false;

    for (int j = 0; j < data.rows(); j++) {
      if (result.row(i) == data.row(j) && result_groups[i] == groups[j]) {
        found = true;
        break;
      }
    }

    ASSERT_TRUE(found) << "Expected to find row [" << result.row(i) << "] in the original data: " << std::endl << data << std::endl;
  }
}

TEST(StatsStratifiedProportionalSample, three_groups_of_equal_size) {
  std::mt19937 generator(0);

  Data<long double> data(9, 3);

  data <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    4.0, 4.0, 4.0,
    5.0, 5.0, 5.0,
    6.0, 6.0, 6.0,
    7.0, 7.0, 7.0,
    8.0, 8.0, 8.0,
    9.0, 9.0, 9.0;

  DataColumn<int> groups(9);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2,
    2;

  auto [result, result_groups] = stratified_proportional_sample(data, groups, { 0, 1, 2 }, 6, generator);

  std::map<int, int> result_sizes;

  for (int i = 0; i < result_groups.size(); i++) {
    result_sizes[result_groups[i]]++;
  }

  ASSERT_EQ(6, result.rows());
  ASSERT_EQ(2, result_sizes[0]);
  ASSERT_EQ(2, result_sizes[1]);
  ASSERT_EQ(2, result_sizes[2]);

  for (int i = 0; i < result.rows(); i++) {
    bool found = false;

    for (int j = 0; j < data.rows(); j++) {
      if (result.row(i) == data.row(j) && result_groups[i] == groups[j]) {
        found = true;
        break;
      }
    }

    ASSERT_TRUE(found) << "Expected to find row [" << result.row(i) << "] in the original data: " << std::endl << data << std::endl;
  }
}

TEST(StatsStratifiedProportionalSample, two_groups_of_different_size_even) {
  std::mt19937 generator(0);

  Data<long double> data(9, 3);

  data <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    4.0, 4.0, 4.0,
    5.0, 5.0, 5.0,
    6.0, 6.0, 6.0,
    7.0, 7.0, 7.0,
    8.0, 8.0, 8.0,
    9.0, 9.0, 9.0;

  DataColumn<int> groups(9);
  groups <<
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1;

  auto [result, result_groups] = stratified_proportional_sample(data, groups, { 0, 1 }, 6, generator);

  std::map<int, int> result_sizes;

  for (int i = 0; i < result_groups.size(); i++) {
    result_sizes[result_groups[i]]++;
  }

  ASSERT_EQ(6, result.rows());
  ASSERT_EQ(2, result_sizes[0]);
  ASSERT_EQ(4, result_sizes[1]);

  for (int i = 0; i < result.rows(); i++) {
    bool found = false;

    for (int j = 0; j < data.rows(); j++) {
      if (result.row(i) == data.row(j) && result_groups[i] == groups[j]) {
        found = true;
        break;
      }
    }

    ASSERT_TRUE(found) << "Expected to find row [" << result.row(i) << "] in the original data: " << std::endl << data << std::endl;
  }
}

TEST(StatsStratifiedProportionalSample, two_groups_of_different_size_odd) {
  std::mt19937 generator(0);

  Data<long double> data(9, 3);

  data <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    4.0, 4.0, 4.0,
    5.0, 5.0, 5.0,
    6.0, 6.0, 6.0,
    7.0, 7.0, 7.0,
    8.0, 8.0, 8.0,
    9.0, 9.0, 9.0;

  DataColumn<int> groups(9);
  groups <<
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1;

  auto [result, result_groups] = stratified_proportional_sample(data, groups, { 0, 1 }, 5, generator);

  std::map<int, int> result_sizes;

  for (int i = 0; i < result_groups.size(); i++) {
    result_sizes[result_groups[i]]++;
  }

  ASSERT_EQ(5, result.rows());
  ASSERT_EQ(1, result_sizes[0]);
  ASSERT_EQ(4, result_sizes[1]);

  for (int i = 0; i < result.rows(); i++) {
    bool found = false;

    for (int j = 0; j < data.rows(); j++) {
      if (result.row(i) == data.row(j) && result_groups[i] == groups[j]) {
        found = true;
        break;
      }
    }

    ASSERT_TRUE(found) << "Expected to find row [" << result.row(i) << "] in the original data: " << std::endl << data << std::endl;
  }
}
