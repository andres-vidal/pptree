#include <gtest/gtest.h>

#include "GroupSpec.hpp"

using namespace models::stats;


TEST(GroupSpec, GroupSize) {
  Data<float> x(6, 3);
  x <<
    2, 2, 2,
    4, 4, 4,
    1, 1, 1,
    6, 6, 6,
    3, 3, 3,
    5, 5, 5;

  DataColumn<int> y(6);
  y <<
    1,
    1,
    2,
    2,
    3,
    3;

  GroupSpec<float, int> spec(x, y);

  ASSERT_EQ(2, spec.group_size(1));
  ASSERT_EQ(2, spec.group_size(2));
  ASSERT_EQ(2, spec.group_size(3));
}

TEST(GroupSpec, GroupStart) {
  Data<float> x(6, 3);
  x <<
    2, 2, 2,
    4, 4, 4,
    1, 1, 1,
    6, 6, 6,
    3, 3, 3,
    5, 5, 5;

  DataColumn<int> y(6);
  y <<
    1,
    1,
    2,
    2,
    3,
    3;

  GroupSpec<float, int> spec(x, y);

  ASSERT_EQ(0, spec.group_start(1));
  ASSERT_EQ(2, spec.group_start(2));
  ASSERT_EQ(4, spec.group_start(3));
}

TEST(GroupSpec, GroupEnd) {
  Data<float> x(6, 3);
  x <<
    2, 2, 2,
    4, 4, 4,
    1, 1, 1,
    6, 6, 6,
    3, 3, 3,
    5, 5, 5;

  DataColumn<int> y(6);
  y <<
    1,
    1,
    2,
    2,
    3,
    3;

  GroupSpec<float, int> spec(x, y);

  ASSERT_EQ(1, spec.group_end(1));
  ASSERT_EQ(3, spec.group_end(2));
  ASSERT_EQ(5, spec.group_end(3));
}

TEST(GroupSpec, Group) {
  Data<float> x(6, 3);
  x <<
    2, 2, 2,
    4, 4, 4,
    1, 1, 1,
    6, 6, 6,
    3, 3, 3,
    5, 5, 5;


  DataColumn<int> y(6);
  y <<
    1,
    1,
    2,
    2,
    3,
    3;

  GroupSpec<float, int> spec(x, y);

  Data<float> actual = spec.group(1);

  Data<float> expected(2, 3);
  expected <<
    2, 2, 2,
    4, 4, 4;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);

  actual = spec.group(2);

  expected.resize(2, 3);
  expected <<
    1, 1, 1,
    6, 6, 6;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);

  actual = spec.group(3);

  expected.resize(2, 3);
  expected <<
    3, 3, 3,
    5, 5, 5;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupSpec, ErrorGroupsNotContiguous) {
  Data<float> x(6, 3);
  x <<
    2, 2, 2,
    4, 4, 4,
    1, 1, 1,
    6, 6, 6,
    3, 3, 3,
    5, 5, 5;

  DataColumn<int> y(6);
  y <<
    1,
    2,
    3,
    1,
    2,
    3;

  ASSERT_THROW((GroupSpec<float, int>(x, y)), std::invalid_argument);
}

TEST(GroupSpec, Subset) {
  Data<float> x(6, 3);
  x <<
    1, 2, 2,
    1, 4, 4,
    2, 1, 1,
    2, 6, 6,
    3, 3, 3,
    3, 5, 5;

  DataColumn<int> y(6);
  y <<
    1,
    1,
    2,
    2,
    3,
    3;

  GroupSpec<float, int> spec = GroupSpec(x, y).subset({ 1, 3 });

  ASSERT_EQ(2, spec.group_size(1));
  ASSERT_EQ(0, spec.group_start(1));
  ASSERT_EQ(1, spec.group_end(1));

  ASSERT_EQ(2, spec.group_size(3));
  ASSERT_EQ(4, spec.group_start(3));
  ASSERT_EQ(5, spec.group_end(3));

  Data<float> expected_group_1(2, 3);
  expected_group_1 <<
    1, 2, 2,
    1, 4, 4;

  ASSERT_EQ(expected_group_1, spec.group(1));

  Data<float> expected_group_3(2, 3);
  expected_group_3 <<
    3, 3, 3,
    3, 5, 5;

  Data<float> expected_x(4, 3);
  expected_x <<
    1, 2, 2,
    1, 4, 4,
    3, 3, 3,
    3, 5, 5;

  ASSERT_EQ(expected_x, spec.data());
  ASSERT_EQ(4, spec.rows());
  ASSERT_EQ(3, spec.cols());

  DataColumn<float> expected_mean(3);
  expected_mean <<
    2.0,
    3.5,
    3.5;

  ASSERT_EQ(expected_mean, spec.mean());
}
