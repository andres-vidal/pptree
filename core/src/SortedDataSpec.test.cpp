#include <gtest/gtest.h>

#include "SortedDataSpec.hpp"

using namespace models::stats;

TEST(SortedDataSpec, Constructor) {
  Data<float> x(6, 3);
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

  SortedDataSpec<float, int> actual(x, y);

  Data<float> expected_x(6, 3);

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

  std::set<int> expected_classes       = { 1, 2, 3 };
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
  Data<float> x(6, 3);
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

  SortedDataSpec<float, int> actual(x, y);

  ASSERT_EQ(2, actual.group_size(1));
  ASSERT_EQ(2, actual.group_size(2));
  ASSERT_EQ(2, actual.group_size(3));
}

TEST(SortedDataSpec, Group) {
  Data<float> x(6, 3);
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

  SortedDataSpec<float, int> data(x, y);

  Data<float> actual = data.group(1);

  Data<float> expected(2, 3);
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

TEST(SortedDataSpec, Analog) {
  Data<float> x(6, 3);
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

  SortedDataSpec<float, int> data(x, y);

  Data<float> new_x(6, 3);

  new_x <<
    2, 2, 2,
    4, 4, 4,
    1, 1, 1,
    6, 6, 6,
    3, 3, 3,
    5, 5, 5;

  SortedDataSpec<float, int> actual = data.analog(new_x);

  ASSERT_EQ(new_x.size(), actual.x.size());
  ASSERT_EQ(new_x.rows(), actual.x.rows());
  ASSERT_EQ(new_x.cols(), actual.x.cols());
  ASSERT_EQ(new_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(SortedDataSpec, Remap) {
  Data<float> x(6, 3);
  x <<
    1, 1, 1,
    1, 2, 2,
    2, 1, 1,
    2, 2, 2,
    3, 1, 1,
    3, 2, 2;

  DataColumn<int> y(6);
  y <<
    1,
    1,
    2,
    2,
    3,
    3;

  SortedDataSpec<float, int> data(x, y);

  std::map<int, int> mapping = {
    { 1, 0 },
    { 2, 1 },
    { 3, 0 }
  };

  SortedDataSpec<float, int> actual = data.remap(mapping);

  Data<float> new_x(6, 3);
  new_x <<
    1, 1, 1,
    1, 2, 2,
    3, 1, 1,
    3, 2, 2,
    2, 1, 1,
    2, 2, 2;


  DataColumn<int> new_y(6);
  new_y <<
    0,
    0,
    0,
    0,
    1,
    1;

  ASSERT_EQ(new_x.size(), actual.x.size());
  ASSERT_EQ(new_x.rows(), actual.x.rows());
  ASSERT_EQ(new_x.cols(), actual.x.cols());
  ASSERT_EQ(new_x, actual.x);

  ASSERT_EQ(new_y.size(), actual.y.size());
  ASSERT_EQ(new_y.rows(), actual.y.rows());
  ASSERT_EQ(new_y.cols(), actual.y.cols());
  ASSERT_EQ(new_y, actual.y);

  ASSERT_EQ(std::set<int>({ 0, 1 }), actual.classes);
}

TEST(SortedDataSpec, Subset) {
  Data<float> x(6, 3);
  x <<
    1, 1, 1,
    1, 2, 2,
    2, 1, 1,
    2, 2, 2,
    3, 1, 1,
    3, 2, 2;

  DataColumn<int> y(6);
  y <<
    1,
    1,
    2,
    2,
    3,
    3;

  SortedDataSpec<float, int> data(x, y);

  SortedDataSpec<float, int> actual = data.subset({ 1, 3 });

  Data<float> new_x(4, 3);
  new_x <<
    1, 1, 1,
    1, 2, 2,
    3, 1, 1,
    3, 2, 2;

  DataColumn<int> new_y(4);
  new_y <<
    1,
    1,
    3,
    3;

  ASSERT_EQ(new_x.size(), actual.x.size());
  ASSERT_EQ(new_x.rows(), actual.x.rows());
  ASSERT_EQ(new_x.cols(), actual.x.cols());
  ASSERT_EQ(new_x, actual.x);

  ASSERT_EQ(new_y.size(), actual.y.size());
  ASSERT_EQ(new_y.rows(), actual.y.rows());
  ASSERT_EQ(new_y.cols(), actual.y.cols());
  ASSERT_EQ(new_y, actual.y);

  ASSERT_EQ(std::set<int>({ 1, 3 }), actual.classes);
}

TEST(SortedDataSpec, BetweenGroupsSumOfSquaresSingleGroup) {
  Data<float> x(3, 3);
  x <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> y(3);
  y <<
    0,
    0,
    0;

  SortedDataSpec<float, int> data(x, y);

  Data<float> actual = data.bgss();

  Data<float> expected(3, 3);
  expected <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(SortedDataSpec, BetweenGroupsSumOfSquaresTwoEqualGroups) {
  Data<float> x(6, 3);
  x <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0,
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> y(6);
  y <<
    0,
    0,
    0,
    1,
    1,
    1;

  SortedDataSpec<float, int> data(x, y);

  Data<float> actual = data.bgss();

  Data<float> expected(3, 3);
  expected <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;


  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(SortedDataSpec, BetweenGroupsSumOfSquaresMultipleGroupsUnivariate) {
  Data<float> x(8, 1);
  x <<
    23.0,
    25.0,
    18.0,
    29.0,
    19.0,
    21.0,
    35.0,
    17.0;

  DataColumn<int> y(8);
  y <<
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2;

  SortedDataSpec<float, int> data(x, y);

  Data<float> actual = data.bgss();

  Data<float> expected(1, 1);
  expected <<
    19.875;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(SortedDataSpec, BetweenGroupsSumOfSquaresMultipleGroupsUnivariateNonSequentialGroups) {
  Data<float> x(8, 1);
  x <<
    23.0,
    25.0,
    18.0,
    29.0,
    19.0,
    21.0,
    35.0,
    17.0;

  DataColumn<int> y(8);
  y <<
    1,
    1,
    1,
    7,
    7,
    7,
    3,
    3;

  SortedDataSpec<float, int> data(x, y);

  Data<float> actual = data.bgss();

  Data<float> expected(1, 1);
  expected <<
    19.875;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(SortedDataSpec, BetweenGroupsSumOfSquaresMultipleGroupsMultivariate) {
  Data<float> x(8, 3);
  x <<
    23.0, 1.0, 1.0,
    25.0, 1.0, 1.0,
    18.0, 1.0, 1.0,
    29.0, 1.0, 1.0,
    19.0, 1.0, 1.0,
    21.0, 1.0, 1.0,
    35.0, 1.0, 1.0,
    17.0, 1.0, 1.0;

  DataColumn<int> y(8);
  y <<
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2;

  SortedDataSpec<float, int> data(x, y);

  Data<float> actual = data.bgss();

  Data<float> expected(3, 3);
  expected <<
    19.875, 0.0, 0.0,
    0.0,    0.0, 0.0,
    0.0,    0.0, 0.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(SortedDataSpec, WithinGroupsSumOfSquaresSingleGroupNoVariance) {
  Data<float> x(3, 3);
  x <<
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0;

  DataColumn<int> y(3);
  y <<
    0,
    0,
    0;

  SortedDataSpec<float, int> data(x, y);

  Data<float> actual = data.wgss();

  Data<float> expected(3, 3);
  expected <<
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(SortedDataSpec, WithinGroupsSumOfSquaresSingleGroupWithVariance) {
  Data<float> x(3, 3);
  x <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0;

  DataColumn<int> y(3);
  y <<
    0,
    0,
    0;

  SortedDataSpec<float, int> data(x, y);

  Data<float> actual = data.wgss();

  Data<float> expected(3, 3);
  expected <<
    2.0, 2.0, 2.0,
    2.0, 2.0, 2.0,
    2.0, 2.0, 2.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(SortedDataSpec, WithinGroupsSumOfSquaresTwoEqualGroups) {
  Data<float> x(6, 3);
  x <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0;

  DataColumn<int> y(6);
  y <<
    0,
    0,
    0,
    1,
    1,
    1;

  SortedDataSpec<float, int> data(x, y);

  Data<float> actual = data.wgss();

  Data<float> expected(3, 3);
  expected <<
    4.0, 4.0, 4.0,
    4.0, 4.0, 4.0,
    4.0, 4.0, 4.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(SortedDataSpec, WithinGroupsSumOfSquaresTwoGroupsSameVariance) {
  Data<float> x(6, 3);
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

  SortedDataSpec<float, int> data(x, y);

  Data<float> actual = data.wgss();

  Data<float> expected(3, 3);
  expected <<
    4.0, 4.0, 4.0,
    4.0, 4.0, 4.0,
    4.0, 4.0, 4.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(SortedDataSpec, WithinGroupsSumOfSquaresTwoGroupsDifferentVariance) {
  Data<float> x(6, 3);
  x <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    1.0, 1.0, 1.0,
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

  SortedDataSpec<float, int> data(x, y);

  Data<float> actual = data.wgss();

  Data<float> expected(3, 3);
  expected <<
    16.0, 16.0, 16.0,
    16.0, 16.0, 16.0,
    16.0, 16.0, 16.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(SortedDataSpec, WithinGroupsSumOfSquaresMultipleGroupsMultivariate1) {
  Data<float> x(8, 3);
  x <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 2.0, 1.0,
    4.0, 3.0, 2.0,
    5.0, 4.0, 3.0,
    9.0, 8.0, 7.0,
    6.0, 5.0, 4.0;

  DataColumn<int> y(8);
  y <<
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2;


  SortedDataSpec<float, int> data(x, y);

  Data<float> actual = data.wgss();

  Data<float> expected(3, 3);
  expected <<
    24.5, 24.5, 24.5,
    24.5, 24.5, 24.5,
    24.5, 24.5, 24.5;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(SortedDataSpec, WithinGroupsSumOfSquaresMultipleGroupsMultivariate2) {
  Data<float> x(8, 4);
  x <<
    1.0, 2.0, 3.0, 0.0,
    4.0, 5.0, 6.0, 0.0,
    7.0, 8.0, 9.0, 0.0,
    3.0, 2.0, 1.0, 0.0,
    4.0, 3.0, 2.0, 0.0,
    5.0, 4.0, 3.0, 0.0,
    9.0, 8.0, 7.0, 0.0,
    6.0, 5.0, 4.0, 0.0;

  DataColumn<int> y(8);
  y <<
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2;

  SortedDataSpec<float, int> data(x, y);

  Data<float> actual = data.wgss();

  Data<float> expected(4, 4);
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
