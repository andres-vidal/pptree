#include <gtest/gtest.h>

#include "Data.hpp"

using namespace models::stats;

TEST(Data, SelectRowsVectorSingleRow) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::vector<int> indices = { 1 };
  Data<double> actual = select_rows(data, indices);

  Data<double> expected(1, 3);
  expected <<
    4.0, 5.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, SelectRowsVectorMultipleRowsNonAdjacent) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::vector<int> indices = { 0, 2 };
  Data<double> actual = select_rows(data, indices);

  Data<double> expected(2, 3);
  expected <<
    1.0, 2.0, 3.0,
    7.0, 8.0, 9.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, SelectRowsVectorMultipleRowsAdjacent) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::vector<int> indices = { 0, 1 };
  Data<double> actual = select_rows(data, indices);

  Data<double> expected(2, 3);
  expected <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, SelectRowsSetSingleRow) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::set<int> indices = { 1 };
  Data<double> actual = select_rows(data, indices);

  Data<double> expected(1, 3);
  expected <<
    4.0, 5.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, SelectRowsSetMultipleRowsNonAdjacent) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::set<int> indices = { 0, 2 };
  Data<double> actual = select_rows(data, indices);

  Data<double> expected(2, 3);
  expected <<
    1.0, 2.0, 3.0,
    7.0, 8.0, 9.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, SelectRowsSetMultipleRowsAdjacent) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::set<int> indices = { 0, 1 };
  Data<double> actual = select_rows(data, indices);

  Data<double> expected(2, 3);
  expected <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, SelectGroupSingleGroup) {
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

TEST(Data, SelectGroupMultipleGroupsAdjacent) {
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

TEST(Data, SelectGroupMultipleGroupsMixed) {
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

TEST(Data, SelectGroupEmptyResult) {
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

TEST(Data, BetweenGroupsSumOfSquaresSingleGroup) {
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

TEST(Data, BetweenGroupsSumOfSquaresTwoEqualGroups) {
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

TEST(Data, BetweenGroupsSumOfSquaresMultipleGroupsUnivariate) {
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

TEST(Data, BetweenGroupsSumOfSquaresMultipleGroupsUnivariateNonSequentialGroups) {
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
    1,
    1,
    1,
    7,
    7,
    7,
    3,
    3;

  Data<double> actual = between_groups_sum_of_squares(data, groups, { 1, 7, 3 });
  Data<double> expected(1, 1);
  expected <<
    19.875;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, BetweenGroupsSumOfSquaresMultipleGroupsMultivariate) {
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

TEST(Data, WithinGroupsSumOfSquaresSingleGroupNoVariance) {
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

TEST(Data, WithinGroupsSumOfSquaresSingleGroupWithVariance) {
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

TEST(Data, WithinGroupsSumOfSquaresTwoEqualGroups) {
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

TEST(Data, WithinGroupsSumOfSquaresTwoGroupsSameVariance) {
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

TEST(Data, WithinGroupsSumOfSquaresTwoGroupsDifferentVariance) {
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

TEST(Data, WithinGroupsSumOfSquaresMultipleGroupsMultivariate1) {
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

TEST(Data, WithinGroupsSumOfSquaresMultipleGroupsMultivariate2) {
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

TEST(Data, MeanSingleObservation) {
  Data<double> data(1, 3);
  data <<
    1.0, 2.0, 6.0;

  DataColumn<double> actual = mean(data);

  DataColumn<double> expected(3);
  expected <<
    1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, MeanMultipleEqualObservations) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0;

  DataColumn<double> actual = mean(data);

  DataColumn<double> expected(3);
  expected <<
    1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, MeanMultipleDifferentObservations) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<double> actual = mean(data);

  DataColumn<double> expected(3);
  expected <<
    2.0, 3.0, 7.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, CovarianceZeroMatrix) {
  Data<double> data(3, 3);
  data <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  Data<double> result = covariance(data);

  ASSERT_EQ(data.size(), result.size());
  ASSERT_EQ(data.rows(), result.rows());
  ASSERT_EQ(data.cols(), result.cols());
  ASSERT_EQ(data, result);
}

TEST(Data, CovarianceConstantMatrix) {
  Data<double> data(3, 3);
  data <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  Data<double> result = covariance(data);

  Data<double> expected(3, 3);
  expected <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(Data, CovarianceAllConstantColumns) {
  Data<double> data(3, 3);
  data <<
    1, 2, 3,
    1, 2, 3,
    1, 2, 3;

  Data<double> result = covariance(data);

  Data<double> expected(3, 3);
  expected <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(Data, CovarianceSomeConstantColumns) {
  Data<double> data(3, 3);
  data <<
    1, 2, 3,
    1, 3, 3,
    1, 4, 3;

  Data<double> result = covariance(data);

  Data<double> expected(3, 3);
  expected <<
    0, 0, 0,
    0, 1, 0,
    0, 0, 0;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(Data, CovarianceGeneric1) {
  Data<double> data(3, 3);
  data <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  Data<double> result = covariance(data);

  Data<double> expected(3, 3);
  expected <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(Data, CovarianceGeneric2) {
  Data<double> data(3, 3);
  data <<
    1, 2, 3,
    1, 2, 3,
    2, 3, 4;

  Data<double> result = covariance(data);

  Data<double> expected(3, 3);
  expected <<
    0.33333, 0.33333, 0.33333,
    0.33333, 0.33333, 0.33333,
    0.33333, 0.33333, 0.33333;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_TRUE(expected.isApprox(result, 0.0001));
}

TEST(Data, SdZeroMatrix) {
  Data<double> data(3, 3);
  data <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  DataColumn<double> result = sd(data);

  DataColumn<double> expected = DataColumn<double>::Zero(3);

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(Data, SdConstantMatrix) {
  Data<double> data(3, 3);
  data <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  Data<double> result = sd(data);

  DataColumn<double> expected = DataColumn<double>::Zero(3);

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(Data, SdAllConstantColumns) {
  Data<double> data(3, 3);
  data <<
    1, 2, 3,
    1, 2, 3,
    1, 2, 3;

  Data<double> result = sd(data);

  DataColumn<double> expected = DataColumn<double>::Zero(3);

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(Data, SdSomeConstantColumns) {
  Data<double> data(3, 3);
  data <<
    1, 2, 3,
    1, 3, 3,
    1, 4, 3;

  Data<double> result = sd(data);

  DataColumn<double> expected(3);
  expected << 0, 1, 0;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(Data, SdGeneric1) {
  Data<double> data(3, 3);
  data <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  Data<double> result = sd(data);

  DataColumn<double> expected(3);
  expected << 1, 1, 1;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(Data, SdGeneric2) {
  Data<double> data(3, 3);
  data <<
    1, 2, 3,
    1, 2, 3,
    2, 3, 4;

  Data<double> result = sd(data);

  DataColumn<double> expected(3);
  expected << 0.5773503, 0.5773503, 0.5773503;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_TRUE(expected.isApprox(result, 0.000001));
}

TEST(Data, CenterDataSingleObservation) {
  Data<double> data(1, 3);
  data <<
    1.0, 2.0, 6.0;

  Data<double> actual = center(data);

  Data<double> expected = Data<double>::Zero(1, 3);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, CenterDataMultipleEqualObservations) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0;

  Data<double> actual = center(data);

  Data<double> expected = Data<double>::Zero(3, 3);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, CenterDataMultipleDifferentObservations) {
  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  Data<double> actual = center(data);

  Data<double> expected(3, 3);
  expected <<
    -1.0, -1.0, -1.0,
    0.0, 0.0, 0.0,
    1.0, 1.0, 1.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, DescaleDataZeroMatrix) {
  Data<double> data(3, 3);
  data <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  Data<double> actual = descale(data);

  Data<double> expected(3, 3);
  expected <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, DescaleDataConstantMatrix) {
  Data<double> data(3, 3);
  data <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  Data<double> actual = descale(data);

  Data<double> expected(3, 3);
  expected <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, DescaleDataDescaledData) {
  Data<double> data(3, 3);
  data <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  Data<double> actual = descale(data);

  Data<double> expected(3, 3);
  expected <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, DescaleDataScaledData) {
  Data<double> data(3, 3);
  data <<
    2, 4, 6,
    4, 6, 8,
    6, 8, 10;

  Data<double> actual = descale(data);

  Data<double> expected(3, 3);
  expected <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, DescaleDataPartiallyScaledData) {
  Data<double> data(3, 3);
  data <<
    2, 4, 3,
    4, 6, 4,
    6, 8, 5;

  Data<double> actual = descale(data);

  Data<double> expected(3, 3);
  expected <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, ShuffleColumnOfDataFirstColumn) {
  Random::seed(0);

  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  Data<double> shuffled = shuffle_column(data, 0);

  Data<double> expected(3, 3);
  expected <<
    2.0, 2.0, 6.0,
    1.0, 3.0, 7.0,
    3.0, 4.0, 8.0;
  ASSERT_EQ(expected.size(), shuffled.size());
  ASSERT_EQ(expected.rows(), shuffled.rows());
  ASSERT_EQ(expected.cols(), shuffled.cols());
  ASSERT_EQ(expected, shuffled);
}

TEST(Data, ShuffleColumnOfDataMiddleColumn) {
  Random::seed(0);

  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  Data<double> shuffled = shuffle_column(data, 1);

  Data<double> expected(3, 3);
  expected <<
    1.0, 3.0, 6.0,
    2.0, 2.0, 7.0,
    3.0, 4.0, 8.0;

  ASSERT_EQ(expected.size(), shuffled.size());
  ASSERT_EQ(expected.rows(), shuffled.rows());
  ASSERT_EQ(expected.cols(), shuffled.cols());
  ASSERT_EQ(expected, shuffled);
}

TEST(Data, ShuffleColumnOfDataLastColumn) {
  Random::seed(0);

  Data<double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  Data<double> shuffled = shuffle_column(data, 2);

  Data<double> expected(3, 3);
  expected <<
    1.0, 2.0, 7.0,
    2.0, 3.0, 6.0,
    3.0, 4.0, 8.0;

  ASSERT_EQ(expected.size(), shuffled.size());
  ASSERT_EQ(expected.rows(), shuffled.rows());
  ASSERT_EQ(expected.cols(), shuffled.cols());
  ASSERT_EQ(expected, shuffled);
}
