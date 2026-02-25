#include <gtest/gtest.h>

#include "GroupSpec.hpp"
#include "Macros.hpp"

using namespace models::stats;


TEST(GroupSpec, GroupSize) {
  Data<float> x = DATA(float, 6,
      2, 2, 2,
      4, 4, 4,
      1, 1, 1,
      6, 6, 6,
      3, 3, 3,
      5, 5, 5);

  DataColumn<int> y = DATA(int, 6,
      1,
      1,
      2,
      2,
      3,
      3);

  GroupSpec<int> spec(y);

  ASSERT_EQ(2, spec.group_size(1));
  ASSERT_EQ(2, spec.group_size(2));
  ASSERT_EQ(2, spec.group_size(3));
}

TEST(GroupSpec, GroupStart) {
  Data<float> x = DATA(float, 6,
      2, 2, 2,
      4, 4, 4,
      1, 1, 1,
      6, 6, 6,
      3, 3, 3,
      5, 5, 5);

  DataColumn<int> y = DATA(int, 6,
      1,
      1,
      2,
      2,
      3,
      3);

  GroupSpec<int> spec(y);

  ASSERT_EQ(0, spec.group_start(1));
  ASSERT_EQ(2, spec.group_start(2));
  ASSERT_EQ(4, spec.group_start(3));
}

TEST(GroupSpec, GroupEnd) {
  Data<float> x = DATA(float, 6,
      2, 2, 2,
      4, 4, 4,
      1, 1, 1,
      6, 6, 6,
      3, 3, 3,
      5, 5, 5);

  DataColumn<int> y = DATA(int, 6,
      1,
      1,
      2,
      2,
      3,
      3);

  GroupSpec<int> spec(y);

  ASSERT_EQ(1, spec.group_end(1));
  ASSERT_EQ(3, spec.group_end(2));
  ASSERT_EQ(5, spec.group_end(3));
}

TEST(GroupSpec, Group) {
  Data<float> x = DATA(float, 6,
      2, 2, 2,
      4, 4, 4,
      1, 1, 1,
      6, 6, 6,
      3, 3, 3,
      5, 5, 5);


  DataColumn<int> y = DATA(int, 6,
      1,
      1,
      2,
      2,
      3,
      3);

  GroupSpec<int> spec(y);

  Data<float> actual = spec.group(x, 1);

  Data<float> expected = DATA(float, 2,
      2, 2, 2,
      4, 4, 4);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);

  actual = spec.group(x, 2);

  expected = DATA(float, 2,
      1, 1, 1,
      6, 6, 6);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);

  actual = spec.group(x, 3);

  expected = DATA(float, 2,
      3, 3, 3,
      5, 5, 5);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupSpec, ErrorGroupsNotContiguous) {
  Data<float> x = DATA(float, 6,
      2, 2, 2,
      4, 4, 4,
      1, 1, 1,
      6, 6, 6,
      3, 3, 3,
      5, 5, 5);

  DataColumn<int> y = DATA(int, 6,
      1,
      2,
      3,
      1,
      2,
      3);

  ASSERT_THROW((GroupSpec<int>(y)), std::invalid_argument);
}

TEST(GroupSpec, Subset) {
  Data<float> x = DATA(float, 6,
      1, 2, 2,
      1, 4, 4,
      2, 1, 1,
      2, 6, 6,
      3, 3, 3,
      3, 5, 5);

  DataColumn<int> y = DATA(int, 6,
      1,
      1,
      2,
      2,
      3,
      3);

  GroupSpec<int> spec = GroupSpec(y).subset({ 1, 3 });

  ASSERT_EQ(2, spec.group_size(1));
  ASSERT_EQ(0, spec.group_start(1));
  ASSERT_EQ(1, spec.group_end(1));

  ASSERT_EQ(2, spec.group_size(3));
  ASSERT_EQ(4, spec.group_start(3));
  ASSERT_EQ(5, spec.group_end(3));

  Data<float> expected_group_1 = DATA(float, 2,
      1, 2, 2,
      1, 4, 4);

  ASSERT_EQ_DATA(expected_group_1, spec.group(x, 1));

  Data<float> expected_group_3 = DATA(float, 2,
      3, 3, 3,
      3, 5, 5);

  Data<float> expected_x = DATA(float, 4,
      1, 2, 2,
      1, 4, 4,
      3, 3, 3,
      3, 5, 5);

  ASSERT_EQ_DATA(expected_x, spec.data(x));

  DataColumn<float> expected_mean = DATA(float, 3,
      2.0,
      3.5,
      3.5);

  ASSERT_EQ(expected_mean, spec.mean(x));
}

TEST(GroupSpec, Remap) {
  Data<float> x = DATA(float, 6,
      1, 1, 1,
      1, 2, 2,
      2, 1, 1,
      2, 2, 2,
      3, 1, 1,
      3, 2, 2);

  DataColumn<int> y = DATA(int, 6,
      1,
      1,
      2,
      2,
      3,
      3);

  GroupSpec<int> spec(y);

  std::map<int, int> mapping = {
    { 1, 0 },
    { 2, 1 },
    { 3, 0 }
  };

  GroupSpec<int> remapped = spec.remap(mapping);

  Data<float> remapped_x = remapped.data(x);

  ASSERT_EQ_DATA(x, remapped_x);
  ASSERT_EQ(std::set<int>({ 0, 1 }), remapped.groups);
}

TEST(GroupSpec,  BetweenGroupsSumOfSquaresSingleGroup) {
  Data<float> x = DATA(float, 3,
      1.0, 2.0, 6.0,
      2.0, 3.0, 7.0,
      3.0, 4.0, 8.0);

  DataColumn<int> y = DATA(int, 3,
      0,
      0,
      0);

  GroupSpec<int> spec(y);

  Data<float> actual = spec.bgss(x);

  Data<float> expected = DATA(float, 3,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupSpec,  BetweenGroupsSumOfSquaresTwoEqualGroups) {
  Data<float> x = DATA(float, 6,
      1.0, 2.0, 6.0,
      2.0, 3.0, 7.0,
      3.0, 4.0, 8.0,
      1.0, 2.0, 6.0,
      2.0, 3.0, 7.0,
      3.0, 4.0, 8.0);

  DataColumn<int> y = DATA(int, 6,
      0,
      0,
      0,
      1,
      1,
      1);

  GroupSpec<int> spec(y);

  Data<float> actual = spec.bgss(x);

  Data<float> expected = DATA(float, 3,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0);


  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupSpec,  BetweenGroupsSumOfSquaresMultipleGroupsUnivariate) {
  Data<float> x = DATA(float, 8,
      23.0,
      25.0,
      18.0,
      29.0,
      19.0,
      21.0,
      35.0,
      17.0);

  DataColumn<int> y = DATA(int, 8,
      0,
      0,
      0,
      1,
      1,
      1,
      2,
      2);

  GroupSpec<int> spec(y);

  Data<float> actual = spec.bgss(x);

  Data<float> expected = DATA(float, 1,
      19.875);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupSpec,  BetweenGroupsSumOfSquaresMultipleGroupsUnivariateNonSequentialGroups) {
  Data<float> x = DATA(float, 8,
      23.0,
      25.0,
      18.0,
      29.0,
      19.0,
      21.0,
      35.0,
      17.0);

  DataColumn<int> y = DATA(int, 8,
      1,
      1,
      1,
      7,
      7,
      7,
      3,
      3);

  GroupSpec<int> spec(y);

  Data<float> actual = spec.bgss(x);

  Data<float> expected = DATA(float, 1,
      19.875);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupSpec,  BetweenGroupsSumOfSquaresMultipleGroupsMultivariate) {
  Data<float> x = DATA(float, 8,
      23.0, 1.0, 1.0,
      25.0, 1.0, 1.0,
      18.0, 1.0, 1.0,
      29.0, 1.0, 1.0,
      19.0, 1.0, 1.0,
      21.0, 1.0, 1.0,
      35.0, 1.0, 1.0,
      17.0, 1.0, 1.0);

  DataColumn<int> y = DATA(int, 8,
      0,
      0,
      0,
      1,
      1,
      1,
      2,
      2);

  GroupSpec<int> spec(y);

  Data<float> actual = spec.bgss(x);

  Data<float> expected = DATA(float, 3,
      19.875, 0.0, 0.0,
      0.0,    0.0, 0.0,
      0.0,    0.0, 0.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupSpec,  WithinGroupsSumOfSquaresSingleGroupNoVariance) {
  Data<float> x = DATA(float, 3,
      1.0, 1.0, 1.0,
      1.0, 1.0, 1.0,
      1.0, 1.0, 1.0);

  DataColumn<int> y = DATA(int, 3,
      0,
      0,
      0);

  GroupSpec<int> spec(y);

  Data<float> actual = spec.wgss(x);

  Data<float> expected = DATA(float, 3,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupSpec,  WithinGroupsSumOfSquaresSingleGroupWithVariance) {
  Data<float> x = DATA(float, 3,
      1.0, 1.0, 1.0,
      2.0, 2.0, 2.0,
      3.0, 3.0, 3.0);

  DataColumn<int> y = DATA(int, 3,
      0,
      0,
      0);

  GroupSpec<int> spec(y);

  Data<float> actual = spec.wgss(x);

  Data<float> expected = DATA(float, 3,
      2.0, 2.0, 2.0,
      2.0, 2.0, 2.0,
      2.0, 2.0, 2.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupSpec,  WithinGroupsSumOfSquaresTwoEqualGroups) {
  Data<float> x = DATA(float, 6,
      1.0, 1.0, 1.0,
      2.0, 2.0, 2.0,
      3.0, 3.0, 3.0,
      1.0, 1.0, 1.0,
      2.0, 2.0, 2.0,
      3.0, 3.0, 3.0);

  DataColumn<int> y = DATA(int, 6,
      0,
      0,
      0,
      1,
      1,
      1);

  GroupSpec<int> spec(y);

  Data<float> actual = spec.wgss(x);

  Data<float> expected = DATA(float, 3,
      4.0, 4.0, 4.0,
      4.0, 4.0, 4.0,
      4.0, 4.0, 4.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupSpec,  WithinGroupsSumOfSquaresTwoGroupsSameVariance) {
  Data<float> x = DATA(float, 6,
      1.0, 1.0, 1.0,
      2.0, 2.0, 2.0,
      3.0, 3.0, 3.0,
      4.0, 4.0, 4.0,
      5.0, 5.0, 5.0,
      6.0, 6.0, 6.0);

  DataColumn<int> y = DATA(int, 6,
      0,
      0,
      0,
      1,
      1,
      1);

  GroupSpec<int> spec(y);

  Data<float> actual = spec.wgss(x);

  Data<float> expected = DATA(float, 3,
      4.0, 4.0, 4.0,
      4.0, 4.0, 4.0,
      4.0, 4.0, 4.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupSpec,  WithinGroupsSumOfSquaresTwoGroupsDifferentVariance) {
  Data<float> x = DATA(float, 6,
      1.0, 1.0, 1.0,
      2.0, 2.0, 2.0,
      3.0, 3.0, 3.0,
      1.0, 1.0, 1.0,
      5.0, 5.0, 5.0,
      6.0, 6.0, 6.0);

  DataColumn<int> y = DATA(int, 6,
      0,
      0,
      0,
      1,
      1,
      1);

  GroupSpec<int> spec(y);

  Data<float> actual = spec.wgss(x);

  Data<float> expected = DATA(float, 3,
      16.0, 16.0, 16.0,
      16.0, 16.0, 16.0,
      16.0, 16.0, 16.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupSpec,  WithinGroupsSumOfSquaresMultipleGroupsMultivariate1) {
  Data<float> x = DATA(float, 8,
      1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
      7.0, 8.0, 9.0,
      3.0, 2.0, 1.0,
      4.0, 3.0, 2.0,
      5.0, 4.0, 3.0,
      9.0, 8.0, 7.0,
      6.0, 5.0, 4.0);

  DataColumn<int> y = DATA(int, 8,
      0,
      0,
      0,
      1,
      1,
      1,
      2,
      2);


  GroupSpec<int> spec(y);

  Data<float> actual = spec.wgss(x);

  Data<float> expected = DATA(float, 3,
      24.5, 24.5, 24.5,
      24.5, 24.5, 24.5,
      24.5, 24.5, 24.5);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupSpec,  WithinGroupsSumOfSquaresMultipleGroupsMultivariate2) {
  Data<float> x = DATA(float, 8,
      1.0, 2.0, 3.0, 0.0,
      4.0, 5.0, 6.0, 0.0,
      7.0, 8.0, 9.0, 0.0,
      3.0, 2.0, 1.0, 0.0,
      4.0, 3.0, 2.0, 0.0,
      5.0, 4.0, 3.0, 0.0,
      9.0, 8.0, 7.0, 0.0,
      6.0, 5.0, 4.0, 0.0);

  DataColumn<int> y = DATA(int, 8,
      0,
      0,
      0,
      1,
      1,
      1,
      2,
      2);

  GroupSpec<int> spec(y);

  Data<float> actual = spec.wgss(x);

  Data<float> expected = DATA(float, 4,
      24.5, 24.5, 24.5, 0.0,
      24.5, 24.5, 24.5, 0.0,
      24.5, 24.5, 24.5, 0.0,
      0.0,  0.0,  0.0,  0.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupSpecRemapped, Group) {
  Data<float> x = DATA(float, 6,
      2, 2, 2,
      4, 4, 4,
      1, 1, 1,
      6, 6, 6,
      3, 3, 3,
      5, 5, 5);


  DataColumn<int> y = DATA(int, 6,
      1,
      1,
      2,
      2,
      3,
      3);

  GroupSpec<int> base(y);
  GroupSpec<int> spec = base.remap({ { 1, 1 }, { 2, 1 }, { 3, 2 } });

  Data<float> actual = spec.group(x, 1);

  Data<float> expected = DATA(float, 4,
      2, 2, 2,
      4, 4, 4,
      1, 1, 1,
      6, 6, 6);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);

  actual = spec.group(x, 2);

  expected = DATA(float, 2,
      3, 3, 3,
      5, 5, 5);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupSpecRemapped, BetweenGroupsSumOfSquaresMultipleGroupsMultivariate) {
  Data<float> x = DATA(float, 8,
      23.0, 1.0, 1.0,
      25.0, 1.0, 1.0,
      18.0, 1.0, 1.0,
      29.0, 1.0, 1.0,
      19.0, 1.0, 1.0,
      21.0, 1.0, 1.0,
      35.0, 1.0, 1.0,
      17.0, 1.0, 1.0);

  DataColumn<int> y = DATA(int, 8,
      0, // 0
      0, // 0
      1, // 0
      2, // 1
      2, // 1
      3, // 1
      4, // 2
      4); // 2

  GroupSpec<int> spec(y);
  GroupSpec<int> remapped = spec.remap({ { 0, 0 }, { 1, 0 }, { 2, 1 }, { 3, 1 }, { 4, 2 } });

  Data<float> actual = remapped.bgss(x);

  Data<float> expected = DATA(float, 3,
      19.875, 0.0, 0.0,
      0.0,    0.0, 0.0,
      0.0,    0.0, 0.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupSpecRemapped, WithinGroupsSumOfSquaresMultipleGroupsMultivariate) {
  Data<float> x = DATA(float, 8,
      1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
      7.0, 8.0, 9.0,
      3.0, 2.0, 1.0,
      4.0, 3.0, 2.0,
      5.0, 4.0, 3.0,
      9.0, 8.0, 7.0,
      6.0, 5.0, 4.0);

  DataColumn<int> y = DATA(int, 8,
      0, // 0
      0, // 0
      1, // 0
      2, // 1
      2, // 1
      3, // 1
      4, // 2
      4); // 2


  GroupSpec<int> spec(y);

  GroupSpec<int> remapped = spec.remap({ { 0, 0 }, { 1, 0 }, { 2, 1 }, { 3, 1 }, { 4, 2 } });

  Data<float> actual = remapped.wgss(x);

  Data<float> expected = DATA(float, 3,
      24.5, 24.5, 24.5,
      24.5, 24.5, 24.5,
      24.5, 24.5, 24.5);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}
