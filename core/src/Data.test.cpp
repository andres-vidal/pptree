#include <gtest/gtest.h>

#include "Data.hpp"

using namespace models::stats;

TEST(Data, SelectRowsVectorSingleRow) {
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

TEST(Data, SelectRowsVectorMultipleRowsNonAdjacent) {
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

TEST(Data, SelectRowsVectorMultipleRowsAdjacent) {
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

TEST(Data, SelectRowsSetSingleRow) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::set<int> indices = { 1 };
  Data<long double> actual = select_rows(data, indices);

  Data<long double> expected(1, 3);
  expected <<
    4.0, 5.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, SelectRowsSetMultipleRowsNonAdjacent) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::set<int> indices = { 0, 2 };
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

TEST(Data, SelectRowsSetMultipleRowsAdjacent) {
  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::set<int> indices = { 0, 1 };
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

TEST(Data, SelectGroupSingleGroup) {
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

TEST(Data, SelectGroupMultipleGroupsAdjacent) {
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

TEST(Data, SelectGroupMultipleGroupsMixed) {
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

TEST(Data, SelectGroupEmptyResult) {
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

TEST(Data, BetweenGroupsSumOfSquaresSingleGroup) {
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

TEST(Data, BetweenGroupsSumOfSquaresTwoEqualGroups) {
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

TEST(Data, BetweenGroupsSumOfSquaresMultipleGroupsUnivariate) {
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

TEST(Data, BetweenGroupsSumOfSquaresMultipleGroupsUnivariateNonSequentialGroups) {
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

TEST(Data, BetweenGroupsSumOfSquaresMultipleGroupsMultivariate) {
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

TEST(Data, WithinGroupsSumOfSquaresSingleGroupNoVariance) {
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

TEST(Data, WithinGroupsSumOfSquaresSingleGroupWithVariance) {
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

TEST(Data, WithinGroupsSumOfSquaresTwoEqualGroups) {
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

TEST(Data, WithinGroupsSumOfSquaresTwoGroupsSameVariance) {
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

TEST(Data, WithinGroupsSumOfSquaresTwoGroupsDifferentVariance) {
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

TEST(Data, WithinGroupsSumOfSquaresMultipleGroupsMultivariate1) {
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

TEST(Data, WithinGroupsSumOfSquaresMultipleGroupsMultivariate2) {
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

TEST(Data, MaskNullColumnsZeroMatrix) {
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

  ASSERT_EQ(expected_mask, mask);
  ASSERT_EQ(expected_indx, indx);
}

TEST(Data, MaskNullColumnsNoNullColumns) {
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

  ASSERT_EQ(expected_mask, mask);
  ASSERT_EQ(expected_indx, indx);
}

TEST(Data, MaskNullColumnsSomeNullColumns) {
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

  ASSERT_EQ(expected_mask, mask);
  ASSERT_EQ(expected_indx, indx);
}

TEST(Data, MeanSingleObservation) {
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

TEST(Data, MeanMultipleEqualObservations) {
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

TEST(Data, MeanMultipleDifferentObservations) {
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

TEST(Data, CovarianceZeroMatrix) {
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

TEST(Data, CovarianceConstantMatrix) {
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

TEST(Data, CovarianceAllConstantColumns) {
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

TEST(Data, CovarianceSomeConstantColumns) {
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

TEST(Data, CovarianceGeneric1) {
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

TEST(Data, CovarianceGeneric2) {
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

TEST(Data, SdZeroMatrix) {
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

TEST(Data, SdConstantMatrix) {
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

TEST(Data, SdAllConstantColumns) {
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

TEST(Data, SdSomeConstantColumns) {
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

TEST(Data, SdGeneric1) {
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

TEST(Data, SdGeneric2) {
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

TEST(Data, CenterDataSingleObservation) {
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

TEST(Data, CenterDataMultipleEqualObservations) {
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

TEST(Data, CenterDataMultipleDifferentObservations) {
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

TEST(Data, DescaleDataZeroMatrix) {
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

TEST(Data, DescaleDataConstantMatrix) {
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

TEST(Data, DescaleDataDescaledData) {
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

TEST(Data, DescaleDataScaledData) {
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

TEST(Data, DescaleDataPartiallyScaledData) {
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

TEST(Data, ShuffleColumnOfDataFirstColumn) {
  Random::rng.seed(0.0);

  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  Data<long double> shuffled = shuffle_column(data, 0);

  ASSERT_EQ(data.size(), shuffled.size());
  ASSERT_EQ(data.rows(), shuffled.rows());
  ASSERT_EQ(data.cols(), shuffled.cols());
  ASSERT_EQ(data.col(1), shuffled.col(1));
  ASSERT_EQ(data.col(2), shuffled.col(2));

  ASSERT_NE(data.col(0), shuffled.col(0));
}

TEST(Data, ShuffleColumnOfDataMiddleColumn) {
  Random::rng.seed(0.0);

  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  Data<long double> shuffled = shuffle_column(data, 1);

  ASSERT_EQ(data.size(), shuffled.size());
  ASSERT_EQ(data.rows(), shuffled.rows());
  ASSERT_EQ(data.cols(), shuffled.cols());
  ASSERT_EQ(data.col(0), shuffled.col(0));
  ASSERT_EQ(data.col(2), shuffled.col(2));

  ASSERT_NE(data.col(1), shuffled.col(1));
}

TEST(Data, ShuffleColumnOfDataLastColumn) {
  Random::rng.seed(0.0);

  Data<long double> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  Data<long double> shuffled = shuffle_column(data, 2);

  ASSERT_EQ(data.size(), shuffled.size());
  ASSERT_EQ(data.rows(), shuffled.rows());
  ASSERT_EQ(data.cols(), shuffled.cols());
  ASSERT_EQ(data.col(0), shuffled.col(0));
  ASSERT_EQ(data.col(1), shuffled.col(1));

  ASSERT_NE(data.col(2), shuffled.col(2));
}
