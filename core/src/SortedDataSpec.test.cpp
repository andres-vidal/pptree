#include <gtest/gtest.h>

#include "SortedDataSpec.hpp"

using namespace models::stats;

TEST(SortedDataSpec, Constructor) {
  Data<double> x(6, 3);
  x <<
    1, 1, 1,
    2, 2, 2,
    3, 3, 3,
    4, 4, 4,
    5, 5, 5,
    6, 6, 6;


  DataColumn<int> y(6);
  y <<
    2,
    1,
    3,
    1,
    3,
    2;

  SortedDataSpec<double, int> actual(x, y);

  Data<double> expected_x(6, 3);

  expected_x <<
    2, 2, 2,
    4, 4, 4,
    1, 1, 1,
    6, 6, 6,
    3, 3, 3,
    5, 5, 5;

  DataColumn<int> expected_y(6);
  expected_y <<
    1,
    1,
    2,
    2,
    3,
    3;

  std::set<int> expected_classes = { 1, 2, 3 };
  std::vector<int> expected_boundaries = { 1, 3 };

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(expected_y.size(), actual.y.size());
  ASSERT_EQ(expected_y.rows(), actual.y.rows());
  ASSERT_EQ(expected_y.cols(), actual.y.cols());
  ASSERT_EQ(expected_y, actual.y);

  ASSERT_EQ(expected_classes, actual.classes);
}

TEST(SortedDataSpec, GroupSize) {
  Data<double> x(6, 3);
  x <<
    1, 1, 1,
    2, 2, 2,
    3, 3, 3,
    4, 4, 4,
    5, 5, 5,
    6, 6, 6;


  DataColumn<int> y(6);

  y <<
    2,
    1,
    3,
    1,
    3,
    2;

  SortedDataSpec<double, int> actual(x, y);

  ASSERT_EQ(2, actual.group_size(1));
  ASSERT_EQ(2, actual.group_size(2));
  ASSERT_EQ(2, actual.group_size(3));
}

TEST(SortedDataSpec, Group) {
  Data<double> x(6, 3);
  x <<
    1, 1, 1,
    2, 2, 2,
    3, 3, 3,
    4, 4, 4,
    5, 5, 5,
    6, 6, 6;


  DataColumn<int> y(6);

  y <<
    2,
    1,
    3,
    1,
    3,
    2;

  SortedDataSpec<double, int> data(x, y);

  Data<double> actual = data.group(1);

  Data<double> expected(2, 3);
  expected <<
    2, 2, 2,
    4, 4, 4;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);

  actual = data.group(2);

  expected.resize(2, 3);
  expected <<
    1, 1, 1,
    6, 6, 6;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);

  actual = data.group(3);

  expected.resize(2, 3);
  expected <<
    3, 3, 3,
    5, 5, 5;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}
