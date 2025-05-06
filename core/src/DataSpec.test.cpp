#include <gtest/gtest.h>

#include "DataSpec.hpp"

using namespace models::stats;


TEST(DataSpec, GroupSize) {
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

  DataSpec<float, int> spec(x, y);

  ASSERT_EQ(2, spec.group_size(1));
  ASSERT_EQ(2, spec.group_size(2));
  ASSERT_EQ(2, spec.group_size(3));
}

TEST(DataSpec, GroupStart) {
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

  DataSpec<float, int> spec(x, y);

  ASSERT_EQ(0, spec.group_start(1));
  ASSERT_EQ(2, spec.group_start(2));
  ASSERT_EQ(4, spec.group_start(3));
}

TEST(DataSpec, GroupEnd) {
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

  DataSpec<float, int> spec(x, y);

  ASSERT_EQ(1, spec.group_end(1));
  ASSERT_EQ(3, spec.group_end(2));
  ASSERT_EQ(5, spec.group_end(3));
}

TEST(DataSpec, Group) {
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

  DataSpec<float, int> spec(x, y);

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

TEST(DataSpec, ErrorGroupsNotContiguous) {
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

  ASSERT_THROW((DataSpec<float, int>(x, y)), std::invalid_argument);
}

TEST(DataSpec, Subset) {
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

  DataSpec<float, int> spec = DataSpec(x, y).subset({ 1, 3 });

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

TEST(DataSpec, Remap) {
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

  DataSpec<float, int> data(x, y);

  std::map<int, int> mapping = {
    { 1, 0 },
    { 2, 1 },
    { 3, 0 }
  };

  DataSpec<float, int> remapped = data.remap(mapping);

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


  ASSERT_EQ_MATRIX(data.x, remapped.x);

  ASSERT_EQ(std::set<int>({ 0, 1 }), remapped.groups);
}


TEST(DataSpec,  BetweenGroupsSumOfSquaresSingleGroup) {
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

  DataSpec<float, int> data(x, y);

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

TEST(DataSpec,  BetweenGroupsSumOfSquaresTwoEqualGroups) {
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

  DataSpec<float, int> data(x, y);

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

TEST(DataSpec,  BetweenGroupsSumOfSquaresMultipleGroupsUnivariate) {
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

  DataSpec<float, int> data(x, y);

  Data<float> actual = data.bgss();

  Data<float> expected(1, 1);
  expected <<
    19.875;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DataSpec,  BetweenGroupsSumOfSquaresMultipleGroupsUnivariateNonSequentialGroups) {
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

  DataSpec<float, int> data(x, y);

  Data<float> actual = data.bgss();

  Data<float> expected(1, 1);
  expected <<
    19.875;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DataSpec,  BetweenGroupsSumOfSquaresMultipleGroupsMultivariate) {
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

  DataSpec<float, int> data(x, y);

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

TEST(DataSpec,  WithinGroupsSumOfSquaresSingleGroupNoVariance) {
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

  DataSpec<float, int> data(x, y);

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

TEST(DataSpec,  WithinGroupsSumOfSquaresSingleGroupWithVariance) {
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

  DataSpec<float, int> data(x, y);

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

TEST(DataSpec,  WithinGroupsSumOfSquaresTwoEqualGroups) {
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

  DataSpec<float, int> data(x, y);

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

TEST(DataSpec,  WithinGroupsSumOfSquaresTwoGroupsSameVariance) {
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

  DataSpec<float, int> data(x, y);

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

TEST(DataSpec,  WithinGroupsSumOfSquaresTwoGroupsDifferentVariance) {
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

  DataSpec<float, int> data(x, y);

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

TEST(DataSpec,  WithinGroupsSumOfSquaresMultipleGroupsMultivariate1) {
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


  DataSpec<float, int> data(x, y);

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

TEST(DataSpec,  WithinGroupsSumOfSquaresMultipleGroupsMultivariate2) {
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

  DataSpec<float, int> data(x, y);

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

TEST(DataSpecRemapped, Group) {
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

  DataSpec<float, int> base(x, y);
  DataSpec<float, int> spec = base.remap({ { 1, 1 }, { 2, 1 }, { 3, 2 } });

  Data<float> actual = spec.group(1);

  Data<float> expected(4, 3);
  expected <<
    2, 2, 2,
    4, 4, 4,
    1, 1, 1,
    6, 6, 6;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);

  actual = spec.group(2);

  expected.resize(2, 3);
  expected <<
    3, 3, 3,
    5, 5, 5;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DataSpecRemapped, BetweenGroupsSumOfSquaresMultipleGroupsMultivariate) {
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
    0, // 0
    0, // 0
    1, // 0
    2, // 1
    2, // 1
    3, // 1
    4, // 2
    4; // 2

  DataSpec<float, int> data(x, y);
  DataSpec<float, int> remapped = data.remap({ { 0, 0 }, { 1, 0 }, { 2, 1 }, { 3, 1 }, { 4, 2 } });

  Data<float> actual = remapped.bgss();

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

TEST(DataSpecRemapped, WithinGroupsSumOfSquaresMultipleGroupsMultivariate) {
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
    0,  // 0
    0,  // 0
    1,  // 0
    2,  // 1
    2,  // 1
    3,  // 1
    4,  // 2
    4;  // 2


  DataSpec<float, int> data(x, y);

  data.inspect();

  DataSpec<float, int> remapped = data.remap({ { 0, 0 }, { 1, 0 }, { 2, 1 }, { 3, 1 }, { 4, 2 } });

  remapped.inspect();

  Data<float> actual = remapped.wgss();

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
