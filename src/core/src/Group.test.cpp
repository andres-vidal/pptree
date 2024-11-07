#include <gtest/gtest.h>

#include "Group.hpp"

using namespace models::stats;


TEST(Group, BinaryRegroupSingleGroup) {
  Data<double> data(3, 1);
  data <<
    1.0,
    4.0,
    7.0;
  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    1;

  ASSERT_THROW({ binary_regroup(data, groups, { 1 }); }, std::runtime_error);
}

TEST(Group, BinaryRegroupTwoGroups) {
  Data<double> data(3, 1);
  data <<
    1.0,
    4.0,
    7.0;
  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    2;

  ASSERT_THROW({ binary_regroup(data, groups, { 1, 2 }); }, std::runtime_error);
}

TEST(Group, BinaryRegroupMultidimensional) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;
  DataColumn<int> groups(3);
  groups <<
    1,
    2,
    3;

  ASSERT_THROW({ binary_regroup(data, groups, { 1, 2, 3 }); },  std::runtime_error);
}

TEST(Group, BinaryRegroupSingleObservationPerGroup) {
  Data<double> data(3, 1);
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

TEST(Group, BinaryRegroupMultipleObservationsPerGroupAdjacent) {
  Data<double> data(8, 1);
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

TEST(Group, BinaryRegroupMultipleObservationsPerGroupMixed) {
  Data<double> data(8, 1);
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
