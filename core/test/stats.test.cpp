#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "stats.hpp"


using namespace stats;
using namespace Eigen;

#define ASSERT_VEC_EQ(expected, actual) \
        ASSERT_EQ(nlohmann::json(expected).dump(), nlohmann::json(actual).dump())


TEST(StatsSelectDataRows, single_row) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::vector<int> indices = { 1 };
  Data<long double> actual = select_rows(data, indices);

  Data<long double> expected(1, 3);
  expected <<
    4.0, 5.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectDataRows, multiple_rows_non_adjacent) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::vector<int> indices = { 0, 2 };
  Data<long double> actual = select_rows(data, indices);

  Data<long double> expected(2, 3);
  expected <<
    1.0, 2.0, 3.0,
    7.0, 8.0, 9.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectDataRows, multiple_rows_adjacent) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::vector<int> indices = { 0, 1 };
  Data<long double> actual = select_rows(data, indices);

  Data<long double> expected(2, 3);
  expected <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectDataColumnRows, single_row) {
  DataColumn<long double> data(3);
  data << 1.0, 2.0, 3.0;

  std::vector<int> indices = { 1 };
  DataColumn<long double> actual = select_rows(data, indices);

  DataColumn<long double> expected(1);
  expected << 2.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectDataColumnRows, multiple_rows_non_adjacent) {
  DataColumn<long double> data(3);
  data << 1.0, 2.0, 3.0;

  std::vector<int> indices = { 0, 2 };
  DataColumn<long double> actual = select_rows(data, indices);

  DataColumn<long double> expected(2);
  expected << 1.0, 3.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectDataColumnRows, multiple_rows_adjacent) {
  DataColumn<long double> data(3);
  data << 1.0, 2.0, 3.0;

  std::vector<int> indices = { 0, 1 };
  DataColumn<long double> actual = select_rows(data, indices);

  DataColumn<long double> expected(2);
  expected << 1.0, 2.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectGroupIndices, single_group) {
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

  std::vector<int> actual = select_group(groups, 1);
  std::vector<int> expected = { 0, 1, 2 };

  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectGroupIndices, multiple_groups_adjacent) {
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

  std::vector<int> actual = select_group(groups, 1);
  std::vector<int> expected = { 0, 1 };

  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectGroupIndices, multiple_groups_mixed) {
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

  std::vector<int> actual = select_group(groups, 1);
  std::vector<int> expected = { 0, 2 };

  ASSERT_EQ(expected, actual);
}

TEST(StatsSelectGroupIndices, empty_result) {
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

  std::vector<int> actual = select_group(groups, 2);
  std::vector<int> expected = {};

  ASSERT_EQ(expected, actual);
}

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

TEST(StatsSelectGroups, multiple_on_multiple2) {
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

#ifndef NDEBUG

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

#endif // NDEBUG

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

#ifndef NDEBUG

TEST(StatsStratifiedProportionalSample, negative_sample_size) {
  std::mt19937 generator(0);

  Data<long double> x(6, 3);
  x <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> y(6);
  y <<
    0,
    0,
    0,
    1,
    1,
    1;

  DataSpec<long double, int> data(x, y, { 0, 1 });

  ASSERT_DEATH({ stratified_proportional_sample(data, -1, generator); }, "Sample size must be greater than 0.");
}

TEST(StatsStratifiedProportionalSample, zero_sample_size) {
  std::mt19937 generator(0);

  Data<long double> x(6, 3);
  x <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> y(6);
  y <<
    0,
    0,
    0,
    1,
    1,
    1;

  DataSpec<long double, int> data(x, y, { 0, 1 });

  ASSERT_DEATH({ stratified_proportional_sample(data, 0, generator); }, "Sample size must be greater than 0.");
}

TEST(StatsStratifiedProportionalSample, sample_size_larger_than_data_rows) {
  std::mt19937 generator(0);

  Data<long double> x(6, 3);
  x <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> y(6);
  y <<
    0,
    0,
    0,
    1,
    1,
    1;

  DataSpec<long double, int> data(x, y, { 0, 1 });

  ASSERT_DEATH({ stratified_proportional_sample(data, 7, generator); }, "Sample size cannot be larger than the number of rows in the data.");
}

#endif // NDEBUG

TEST(StatsStratifiedProportionalSample, sample_has_correct_size) {
  std::mt19937 generator(0);

  Data<long double> x(6, 3);
  x <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> y(6);
  y <<
    0,
    0,
    0,
    1,
    1,
    1;

  DataSpec<long double, int> data(x, y, { 0, 1 });

  DataSpec<long double, int> result = stratified_proportional_sample(data, 4, generator).get_sample();

  ASSERT_EQ(4, result.x.rows());
  ASSERT_EQ(3, result.x.cols());
  ASSERT_EQ(4, result.y.size());
}

TEST(StatsStratifiedProportionalSample, sample_has_correct_size_per_strata) {
  std::mt19937 generator(0);

  Data<long double> x(6, 3);
  x <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> y(6);
  y <<
    0,
    0,
    0,
    1,
    1,
    1;

  DataSpec<long double, int> data(x, y, { 0, 1 });

  DataSpec<long double, int> result = stratified_proportional_sample(data, 4, generator).get_sample();

  std::map<int, int> result_sizes;

  for (int i = 0; i < result.y.size(); i++) {
    result_sizes[result.y[i]]++;
  }

  ASSERT_EQ(2, result_sizes[0]);
  ASSERT_EQ(2, result_sizes[1]);
}

TEST(StatsStratifiedProportionalSample, sample_is_subset_of_data_per_strata) {
  std::mt19937 generator(0);

  Data<long double> x(6, 3);
  x <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    4.0, 4.0, 4.0,
    5.0, 5.0, 5.0,
    6.0, 6.0, 6.0;

  DataColumn<int> y(6);
  y <<
    0,
    0,
    0,
    1,
    1,
    1;

  DataSpec<long double, int> data(x, y, { 0, 1 });

  DataSpec<long double, int> result = stratified_proportional_sample(data, 4, generator).get_sample();

  for (int i = 0; i < result.x.rows(); i++) {
    bool found = false;

    for (int j = 0; j < data.x.rows(); j++) {
      if (result.x.row(i) == data.x.row(j) && result.y[i] == data.y[j]) {
        found = true;
        break;
      }
    }

    ASSERT_TRUE(found) << "Expected to find row [" << result.x.row(i) << "] in the original data: " << std::endl << data.x << std::endl;
  }
}

TEST(StatsStratifiedProportionalSample, three_groups_of_equal_size) {
  std::mt19937 generator(0);

  Data<long double> x(9, 3);

  x <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    4.0, 4.0, 4.0,
    5.0, 5.0, 5.0,
    6.0, 6.0, 6.0,
    7.0, 7.0, 7.0,
    8.0, 8.0, 8.0,
    9.0, 9.0, 9.0;

  DataColumn<int> y(9);
  y <<
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2,
    2;

  DataSpec<long double, int> data(x, y, { 0, 1, 2 });

  DataSpec<long double, int> result = stratified_proportional_sample(data, 6, generator).get_sample();

  std::map<int, int> result_sizes;

  for (int i = 0; i < result.y.size(); i++) {
    result_sizes[result.y[i]]++;
  }

  ASSERT_EQ(6, result.x.rows());
  ASSERT_EQ(2, result_sizes[0]);
  ASSERT_EQ(2, result_sizes[1]);
  ASSERT_EQ(2, result_sizes[2]);

  for (int i = 0; i < result.x.rows(); i++) {
    bool found = false;

    for (int j = 0; j < data.x.rows(); j++) {
      if (result.x.row(i) == data.x.row(j) && result.y[i] == data.y[j]) {
        found = true;
        break;
      }
    }

    ASSERT_TRUE(found) << "Expected to find row [" << result.x.row(i) << "] in the original data: " << std::endl << data.x << std::endl;
  }
}

TEST(StatsStratifiedProportionalSample, two_groups_of_different_size_even) {
  std::mt19937 generator(0);

  Data<long double> x(9, 3);

  x <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    4.0, 4.0, 4.0,
    5.0, 5.0, 5.0,
    6.0, 6.0, 6.0,
    7.0, 7.0, 7.0,
    8.0, 8.0, 8.0,
    9.0, 9.0, 9.0;

  DataColumn<int> y(9);
  y <<
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1;

  DataSpec<long double, int> data(x, y, { 0, 1 });

  DataSpec<long double, int> result = stratified_proportional_sample(data, 6, generator).get_sample();

  std::map<int, int> result_sizes;

  for (int i = 0; i < result.y.size(); i++) {
    result_sizes[result.y[i]]++;
  }

  ASSERT_EQ(6, result.x.rows());
  ASSERT_EQ(2, result_sizes[0]);
  ASSERT_EQ(4, result_sizes[1]);

  for (int i = 0; i < result.x.rows(); i++) {
    bool found = false;

    for (int j = 0; j < data.x.rows(); j++) {
      if (result.x.row(i) == data.x.row(j) && result.y[i] == data.y[j]) {
        found = true;
        break;
      }
    }

    ASSERT_TRUE(found) << "Expected to find row [" << result.x.row(i) << "] in the original data: " << std::endl << data.x << std::endl;
  }
}

TEST(StatsStratifiedProportionalSample, two_groups_of_different_size_odd) {
  std::mt19937 generator(0);

  Data<long double> x(9, 3);

  x <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    4.0, 4.0, 4.0,
    5.0, 5.0, 5.0,
    6.0, 6.0, 6.0,
    7.0, 7.0, 7.0,
    8.0, 8.0, 8.0,
    9.0, 9.0, 9.0;

  DataColumn<int> y(9);
  y <<
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1;

  DataSpec<long double, int> data(x, y, { 0, 1 });

  DataSpec<long double, int> result = stratified_proportional_sample(data, 5, generator).get_sample();

  std::map<int, int> result_sizes;

  for (int i = 0; i < result.y.size(); i++) {
    result_sizes[result.y[i]]++;
  }

  ASSERT_EQ(5, result.x.rows());
  ASSERT_EQ(1, result_sizes[0]);
  ASSERT_EQ(4, result_sizes[1]);

  for (int i = 0; i < result.x.rows(); i++) {
    bool found = false;

    for (int j = 0; j < data.x.rows(); j++) {
      if (result.x.row(i) == data.x.row(j) && result.y[i] == data.y[j]) {
        found = true;
        break;
      }
    }

    ASSERT_TRUE(found) << "Expected to find row [" << result.x.row(i) << "] in the original data: " << std::endl << data.x << std::endl;
  }
}

TEST(StatsMaskNullColumns, zero_matrix) {
  Data<long double> data(3, 3);
  data <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  auto [mask, indx] = mask_null_columns(data);

  std::vector<int> expected_mask { 0, 0, 0 };
  std::vector<int> expected_indx {  };

  ASSERT_EQ(expected_mask.size(), mask.size());
  ASSERT_EQ(expected_indx.size(), indx.size());

  ASSERT_VEC_EQ(expected_mask, mask);
  ASSERT_VEC_EQ(expected_indx, indx);
}

TEST(StatsMaskNullColumns, no_null_columns) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  auto [mask, indx] = mask_null_columns(data);

  std::vector<int> expected_mask { 1, 1, 1 };
  std::vector<int> expected_indx { 0, 1, 2 };

  ASSERT_EQ(expected_mask.size(), mask.size());
  ASSERT_EQ(expected_indx.size(), indx.size());

  ASSERT_VEC_EQ(expected_mask, mask);
  ASSERT_VEC_EQ(expected_indx, indx);
}

TEST(StatsMaskNullColumns, some_null_columns) {
  Data<long double> data(3, 3);
  data <<
    1.0, 0.0, 3.0,
    4.0, 0.0, 6.0,
    7.0, 0.0, 9.0;

  auto [mask, indx] = mask_null_columns(data);

  std::vector<int> expected_mask { 1, 0, 1 };
  std::vector<int> expected_indx { 0, 2 };

  ASSERT_EQ(expected_mask.size(), mask.size());
  ASSERT_EQ(expected_indx.size(), indx.size());

  ASSERT_VEC_EQ(expected_mask, mask);
  ASSERT_VEC_EQ(expected_indx, indx);
}

TEST(StatsExpandDataColumn, idempotent) {
  DataColumn<long double> data(3);
  data << 1.0, 2.0, 3.0;

  std::vector<int> mask { 1, 1, 1 };

  Data<long double> actual = expand(data, mask);

  ASSERT_EQ(data.size(), actual.size());
  ASSERT_EQ(data.rows(), actual.rows());
  ASSERT_EQ(data.cols(), actual.cols());
  ASSERT_EQ(data, actual);
}

TEST(StatsExpandDataColumn, generic) {
  DataColumn<long double> data(3);
  data << 1.0, 2.0, 3.0;

  std::vector<int> mask { 1, 0, 1, 0, 1 };

  DataColumn<long double> actual = expand(data, mask);

  DataColumn<long double> expected(5);
  expected <<
    1.0, 0.0, 2.0, 0.0, 3.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsDataMean, single_observation) {
  Data<long double> data(1, 3);
  data <<
    1.0, 2.0, 6.0;

  DataColumn<long double> actual = mean(data);

  DataColumn<long double> expected(3);
  expected <<
    1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsDataMean, multiple_equal_observations) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0;

  DataColumn<long double> actual = mean(data);

  DataColumn<long double> expected(3);
  expected <<
    1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsDataMean, multiple_different_observations) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<long double> actual = mean(data);

  DataColumn<long double> expected(3);
  expected <<
    2.0, 3.0, 7.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsDataColumnMean, single_observation) {
  DataColumn<long double> data(1);
  data <<
    1.0;

  long double actual = mean(data);
  long double expected = 1.0;

  ASSERT_EQ(expected, actual);
}

TEST(StatsDataColumnMean, multiple_equal_observations) {
  DataColumn<long double> data(3);
  data <<
    1.0,
    1.0,
    1.0;

  long double actual = mean(data);
  long double expected = 1.0;

  ASSERT_EQ(expected, actual);
}

TEST(StatsDataColumnMean, multiple_different_observations) {
  DataColumn<long double> data(3);
  data <<
    1.0,
    2.0,
    3.0;

  long double actual = mean(data);
  long double expected = 2.0;

  ASSERT_EQ(expected, actual);
}


TEST(StatsCovariance, zero_matrix_results_in_zero_matrix) {
  Data<long double> data(3, 3);
  data <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  Data<long double> result = covariance(data);

  ASSERT_EQ(data.size(), result.size());
  ASSERT_EQ(data.rows(), result.rows());
  ASSERT_EQ(data.cols(), result.cols());
  ASSERT_EQ(data, result);
}

TEST(StatsCovariance, constant_matrix_results_in_zero_matrix) {
  Data<long double> data(3, 3);
  data <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  Data<long double> result = covariance(data);

  Data<long double> expected(3, 3);
  expected <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(LingAlgCovariance, all_constant_columns_result_in_zero_matrix) {
  Data<long double> data(3, 3);
  data <<
    1, 2, 3,
    1, 2, 3,
    1, 2, 3;

  Data<long double> result = covariance(data);

  Data<long double> expected(3, 3);
  expected <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(StatsCovariance, some_constant_columns_result_in_some_zero_columns) {
  Data<long double> data(3, 3);
  data <<
    1, 2, 3,
    1, 3, 3,
    1, 4, 3;

  Data<long double> result = covariance(data);

  Data<long double> expected(3, 3);
  expected <<
    0, 0, 0,
    0, 1, 0,
    0, 0, 0;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(StatsCovariance, generic_1) {
  Data<long double> data(3, 3);
  data <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  Data<long double> result = covariance(data);

  Data<long double> expected(3, 3);
  expected <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(StatsCovariance, generic_2) {
  Data<long double> data(3, 3);
  data <<
    1, 2, 3,
    1, 2, 3,
    2, 3, 4;

  Data<long double> result = covariance(data);

  Data<long double> expected(3, 3);
  expected <<
    0.33333, 0.33333, 0.33333,
    0.33333, 0.33333, 0.33333,
    0.33333, 0.33333, 0.33333;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_TRUE(expected.isApprox(result, 0.0001));
}

TEST(StatsDataSd, zero_matrix_results_in_zero_matrix) {
  Data<long double> data(3, 3);
  data <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  DataColumn<long double> result = sd(data);

  DataColumn<long double> expected = DataColumn<long double>::Zero(3);

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(StatsDataSd, constant_matrix_results_in_zero_matrix) {
  Data<long double> data(3, 3);
  data <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  Data<long double> result = sd(data);

  DataColumn<long double> expected = DataColumn<long double>::Zero(3);

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(StatsDataSd, all_constant_columns_result_in_zero_matrix) {
  Data<long double> data(3, 3);
  data <<
    1, 2, 3,
    1, 2, 3,
    1, 2, 3;

  Data<long double> result = sd(data);

  DataColumn<long double> expected = DataColumn<long double>::Zero(3);

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(StatsDataSd, some_constant_columns_result_in_some_zero_columns) {
  Data<long double> data(3, 3);
  data <<
    1, 2, 3,
    1, 3, 3,
    1, 4, 3;

  Data<long double> result = sd(data);

  DataColumn<long double> expected(3);
  expected << 0, 1, 0;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(StatsDataSd, generic_1) {
  Data<long double> data(3, 3);
  data <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  Data<long double> result = sd(data);

  DataColumn<long double> expected(3);
  expected << 1, 1, 1;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(StatsDataSd, generic_2) {
  Data<long double> data(3, 3);
  data <<
    1, 2, 3,
    1, 2, 3,
    2, 3, 4;

  Data<long double> result = sd(data);

  DataColumn<long double> expected(3);
  expected << 0.5773503, 0.5773503, 0.5773503;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_TRUE(expected.isApprox(result, 0.000001));
}

TEST(StatsDataColumnSd, zero_vector_results_in_zero) {
  DataColumn<long double> data(3);
  data <<
    0,
    0,
    0;

  long double result = sd(data);
  long double expected = 0;

  ASSERT_EQ(expected, result);
}

TEST(StatsDataColumnSd, constant_vector_results_in_zero) {
  DataColumn<long double> data(3);
  data <<
    1,
    1,
    1;

  long double result = sd(data);
  long double expected = 0;

  ASSERT_EQ(expected, result);
}

TEST(StatsDataColumnSd, generic_1) {
  DataColumn<long double> data(3);
  data <<
    1,
    2,
    3;

  long double result = sd(data);
  long double expected = 1;

  ASSERT_EQ(expected, result);
}

TEST(StatsDataColumnSd, generic_2) {
  DataColumn<long double> data(3);
  data <<
    1,
    1,
    2;

  long double result = sd(data);
  long double expected = 0.5773503;

  ASSERT_NEAR(expected, result, 0.00001);
}

TEST(StatsCenterData, single_observation) {
  Data<long double> data(1, 3);
  data <<
    1.0, 2.0, 6.0;

  Data<long double> actual = center(data);

  Data<long double> expected = Data<long double>::Zero(1, 3);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsCenterData, multiple_equal_observations) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0;

  Data<long double> actual = center(data);

  Data<long double> expected = Data<long double>::Zero(3, 3);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsCenterData, multiple_different_observations) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  Data<long double> actual = center(data);

  Data<long double> expected(3, 3);
  expected <<
    -1.0, -1.0, -1.0,
    0.0, 0.0, 0.0,
    1.0, 1.0, 1.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsCenterDataSpec, single_observation) {
  Data<long double> x(1, 3);
  x <<
    1.0, 2.0, 6.0;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<long double, int> data(x, y);

  DataSpec<long double, int> actual = center(data);

  Data<long double> expected_x = Data<long double>::Zero(1, 3);

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(StatsCenterDataSpec, multiple_equal_observations) {
  Data<long double> x(3, 3);
  x <<
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<long double, int> data(x, y);

  DataSpec<long double, int> actual = center(data);

  Data<long double> expected_x = Data<long double>::Zero(3, 3);

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(StatsCenterDataSpec, multiple_different_observations) {
  Data<long double> x(3, 3);
  x <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<long double, int> data(x, y);

  DataSpec<long double, int> actual = center(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    -1.0, -1.0, -1.0,
    0.0, 0.0, 0.0,
    1.0, 1.0, 1.0;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(StatsCenterBootstrapDataSpec, single_observation) {
  Data<long double> x(1, 3);
  x <<
    1.0, 2.0, 6.0;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  BootstrapDataSpec<long double, int> actual = center(data);

  Data<long double> expected_x = Data<long double>::Zero(1, 3);

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.indices, actual.indices);
  ASSERT_EQ(data.classes, actual.classes);
}

TEST(StatsCenterBootstrapDataSpec, multiple_equal_observations) {
  Data<long double> x(3, 3);
  x <<
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  BootstrapDataSpec<long double, int> actual = center(data);

  Data<long double> expected_x = Data<long double>::Zero(3, 3);

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.indices, actual.indices);
  ASSERT_EQ(data.classes, actual.classes);
}

TEST(StatsCenterBootstrapDataSpec, multiple_different_observations) {
  Data<long double> x(3, 3);
  x <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  BootstrapDataSpec<long double, int> actual = center(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    -1.0, -1.0, -1.0,
    0.0, 0.0, 0.0,
    1.0, 1.0, 1.0;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.indices, actual.indices);
  ASSERT_EQ(data.classes, actual.classes);
}

TEST(StatsCenterDataColumn, single_observation) {
  DataColumn<long double> data(1);
  data << 1.0;

  DataColumn<long double> actual = center(data);

  DataColumn<long double> expected = DataColumn<long double>::Zero(1);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsCenterDataColumn, multiple_equal_observations) {
  DataColumn<long double> data(3);
  data <<
    1.0,
    1.0,
    1.0;

  DataColumn<long double> actual = center(data);

  DataColumn<long double> expected = DataColumn<long double>::Zero(3);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsCenterDataColumn, multiple_different_observations) {
  DataColumn<long double> data(3);
  data <<
    1.0,
    2.0,
    3.0;

  DataColumn<long double> actual = center(data);

  DataColumn<long double> expected(3);
  expected <<
    -1.0,
    0.0,
    1.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsDescaleData, idempotent_in_zero_matrix) {
  Data<long double> data(3, 3);
  data <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  Data<long double> actual = descale(data);

  Data<long double> expected(3, 3);
  expected <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsDescaleData, idempotent_in_constant_matrix) {
  Data<long double> data(3, 3);
  data <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  Data<long double> actual = descale(data);

  Data<long double> expected(3, 3);
  expected <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsDescaleData, idempotent_in_descaled_data) {
  Data<long double> data(3, 3);
  data <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  Data<long double> actual = descale(data);

  Data<long double> expected(3, 3);
  expected <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsDescaleData, descales_scaled_data) {
  Data<long double> data(3, 3);
  data <<
    2, 4, 6,
    4, 6, 8,
    6, 8, 10;

  Data<long double> actual = descale(data);

  Data<long double> expected(3, 3);
  expected <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsDescaleData, descales_partially_scaled_data) {
  Data<long double> data(3, 3);
  data <<
    2, 4, 3,
    4, 6, 4,
    6, 8, 5;

  Data<long double> actual = descale(data);

  Data<long double> expected(3, 3);
  expected <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsDescaleDataColumn, idempotent_in_zero_vector) {
  DataColumn<long double> data(3);
  data <<
    0,
    0,
    0;

  DataColumn<long double> actual = descale(data);

  DataColumn<long double> expected(3);
  expected <<
    0,
    0,
    0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsDescaleDataColumn, idempotent_in_constant_vector) {
  DataColumn<long double> data(3);
  data <<
    1,
    1,
    1;

  DataColumn<long double> actual = descale(data);

  DataColumn<long double> expected(3);
  expected <<
    1,
    1,
    1;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsDescaleDataColumn, idempotent_in_descaled_data) {
  DataColumn<long double> data(3);
  data <<
    1,
    2,
    3;

  DataColumn<long double> actual = descale(data);

  DataColumn<long double> expected(3);
  expected <<
    1,
    2,
    3;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsDescaleDataColumn, descales_scaled_data) {
  DataColumn<long double> data(3);
  data <<
    2,
    4,
    6;

  DataColumn<long double> actual = descale(data);

  DataColumn<long double> expected(3);
  expected <<
    1,
    2,
    3;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(StatsDescaleDataSpec, idempotent_in_zero_matrix) {
  Data<long double> x(3, 3);
  x <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<long double, int> data(x, y);

  DataSpec<long double, int> actual = descale(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(StatsDescaleDataSpec, idempotent_in_constant_matrix) {
  Data<long double> x(3, 3);
  x <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<long double, int> data(x, y);

  DataSpec<long double, int> actual = descale(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(StatsDescaleDataSpec, idempotent_in_descaled_data) {
  Data<long double> x(3, 3);
  x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<long double, int> data(x, y);

  DataSpec<long double, int> actual = descale(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(StatsDescaleDataSpec, descales_scaled_data) {
  Data<long double> x(3, 3);
  x <<
    2, 4, 6,
    4, 6, 8,
    6, 8, 10;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<long double, int> data(x, y);

  DataSpec<long double, int> actual = descale(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(StatsDescaleDataSpec, descales_partially_scaled_data) {
  Data<long double> x(3, 3);
  x <<
    2, 4, 3,
    4, 6, 4,
    6, 8, 5;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<long double, int> data(x, y);

  DataSpec<long double, int> actual = descale(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(StatsDescaleBootstrapDataSpec, idempotent_in_zero_matrix) {
  Data<long double> x(3, 3);
  x <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  BootstrapDataSpec<long double, int> actual = descale(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.indices, actual.indices);
  ASSERT_EQ(data.classes, actual.classes);
}

TEST(StatsDescaleBootstrapDataSpec, idempotent_in_constant_matrix) {
  Data<long double> x(3, 3);
  x <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  BootstrapDataSpec<long double, int> actual = descale(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.indices, actual.indices);
  ASSERT_EQ(data.classes, actual.classes);
}

TEST(StatsDescaleBootstrapDataSpec, idempotent_in_descaled_data) {
  Data<long double> x(3, 3);
  x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  BootstrapDataSpec<long double, int> actual = descale(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.indices, actual.indices);
  ASSERT_EQ(data.classes, actual.classes);
}

TEST(StatsDescaleBootstrapDataSpec, descales_scaled_data) {
  Data<long double> x(3, 3);
  x <<
    2, 4, 6,
    4, 6, 8,
    6, 8, 10;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  BootstrapDataSpec<long double, int> actual = descale(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.indices, actual.indices);
  ASSERT_EQ(data.classes, actual.classes);
}

TEST(StatsDescaleBootstrapDataSpec, descales_partially_scaled_data) {
  Data<long double> x(3, 3);
  x <<
    2, 4, 3,
    4, 6, 4,
    6, 8, 5;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  BootstrapDataSpec<long double, int> actual = descale(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.indices, actual.indices);
  ASSERT_EQ(data.classes, actual.classes);
}


TEST(DataSpecUnwrap, returns_data_groups_and_classes) {
  Data<long double> x(3, 3);
  x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<long double, int> data(x, y);

  auto [unwrapped_x, unwrapped_y, unwrapped_classes] = data.unwrap();

  Data<long double> expected_x(3, 3);
  expected_x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> expected_y(3);
  expected_y <<
    1,
    2,
    3;

  std::set<int> expected_classes = { 1, 2, 3 };

  ASSERT_EQ(expected_x.size(), unwrapped_x.size());
  ASSERT_EQ(expected_x.rows(), unwrapped_x.rows());
  ASSERT_EQ(expected_x.cols(), unwrapped_x.cols());
  ASSERT_EQ(expected_x, unwrapped_x);

  ASSERT_EQ(expected_y.size(), unwrapped_y.size());
  ASSERT_EQ(expected_y.rows(), unwrapped_y.rows());
  ASSERT_EQ(expected_y.cols(), unwrapped_y.cols());
  ASSERT_EQ(expected_y, unwrapped_y);

  ASSERT_EQ(expected_classes.size(), unwrapped_classes.size());
  ASSERT_EQ(expected_classes, unwrapped_classes);
}

TEST(BootstrapDataSpecGetSample, returns_data_spec_from_sampled_indices) {
  Data<long double> x(3, 3);
  x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  DataSpec<long double, int> sample = data.get_sample();

  Data<long double> expected_x(2, 3);
  expected_x <<
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> expected_y(2);
  expected_y <<
    2,
    3;

  std::set<int> expected_classes = { 2, 3 };

  ASSERT_EQ(expected_x.size(), sample.x.size());
  ASSERT_EQ(expected_x.rows(), sample.x.rows());
  ASSERT_EQ(expected_x.cols(), sample.x.cols());
  ASSERT_EQ(expected_x, sample.x);

  ASSERT_EQ(expected_y.size(), sample.y.size());
  ASSERT_EQ(expected_y.rows(), sample.y.rows());
  ASSERT_EQ(expected_y.cols(), sample.y.cols());
  ASSERT_EQ(expected_y, sample.y);

  ASSERT_EQ(expected_classes.size(), sample.classes.size());
  ASSERT_EQ(expected_classes, sample.classes);
}

TEST(BootstrapDataSpecUnwrap, returns_data_groups_and_classes_from_sampled_indices) {
  Data<long double> x(3, 3);
  x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  auto [unwrapped_x, unwrapped_y, unwrapped_classes] = data.unwrap();

  Data<long double> expected_x(2, 3);
  expected_x <<
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> expected_y(2);
  expected_y <<
    2,
    3;

  std::set<int> expected_classes = { 2, 3 };

  ASSERT_EQ(expected_x.size(), unwrapped_x.size());
  ASSERT_EQ(expected_x.rows(), unwrapped_x.rows());
  ASSERT_EQ(expected_x.cols(), unwrapped_x.cols());
  ASSERT_EQ(expected_x, unwrapped_x);

  ASSERT_EQ(expected_y.size(), unwrapped_y.size());
  ASSERT_EQ(expected_y.rows(), unwrapped_y.rows());
  ASSERT_EQ(expected_y.cols(), unwrapped_y.cols());
  ASSERT_EQ(expected_y, unwrapped_y);

  ASSERT_EQ(expected_classes.size(), unwrapped_classes.size());
  ASSERT_EQ(expected_classes, unwrapped_classes);
}

TEST(BootstrapDataSpecUnwrap, repeats_data_rows_for_repeated_indices) {
  Data<long double> x(3, 3);
  x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5,
    4, 5, 6;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3,
    4;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2, 2 });

  auto [unwrapped_x, unwrapped_y, unwrapped_classes] = data.unwrap();

  Data<long double> expected_x(3, 3);
  expected_x <<
    2, 3, 4,
    3, 4, 5,
    3, 4, 5;

  DataColumn<int> expected_y(3);
  expected_y <<
    2,
    3,
    3;

  std::set<int> expected_classes = { 2, 3 };

  ASSERT_EQ(expected_x.size(), unwrapped_x.size());
  ASSERT_EQ(expected_x.rows(), unwrapped_x.rows());
  ASSERT_EQ(expected_x.cols(), unwrapped_x.cols());
  ASSERT_EQ(expected_x, unwrapped_x);

  ASSERT_EQ(expected_y.size(), unwrapped_y.size());
  ASSERT_EQ(expected_y.rows(), unwrapped_y.rows());
  ASSERT_EQ(expected_y.cols(), unwrapped_y.cols());
  ASSERT_EQ(expected_y, unwrapped_y);

  ASSERT_EQ(expected_classes.size(), unwrapped_classes.size());
  ASSERT_EQ(expected_classes, unwrapped_classes);
}
