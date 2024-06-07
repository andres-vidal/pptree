#include <gtest/gtest.h>

#include "DataColumn.hpp"

using namespace models::stats;

TEST(DataColumn, SelectRowsVectorSingleRow) {
  DataColumn<long double> data(3);
  data << 1.0, 2.0, 3.0;

  std::vector<int> indices = { 1 };
  DataColumn<long double> actual = select_rows(data, indices);

  DataColumn<long double> expected(1);
  expected << 2.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, SelectRowsVectorMultipleRowsNonAdjacent) {
  DataColumn<long double> data(3);
  data << 1.0, 2.0, 3.0;

  std::vector<int> indices = { 0, 2 };
  DataColumn<long double> actual = select_rows(data, indices);

  DataColumn<long double> expected(2);
  expected << 1.0, 3.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, SelectRowsVectorMultipleRowsAdjacent) {
  DataColumn<long double> data(3);
  data << 1.0, 2.0, 3.0;

  std::vector<int> indices = { 0, 1 };
  DataColumn<long double> actual = select_rows(data, indices);

  DataColumn<long double> expected(2);
  expected << 1.0, 2.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, SelectRowsSetSingleRow) {
  DataColumn<long double> data(3);
  data << 1.0, 2.0, 3.0;

  std::set<int> indices = { 1 };
  DataColumn<long double> actual = select_rows(data, indices);

  DataColumn<long double> expected(1);
  expected << 2.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, SelectRowsSetMultipleRowsNonAdjacent) {
  DataColumn<long double> data(3);
  data << 1.0, 2.0, 3.0;

  std::set<int> indices = { 0, 2 };
  DataColumn<long double> actual = select_rows(data, indices);

  DataColumn<long double> expected(2);
  expected << 1.0, 3.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, SelectRowsSetMultipleRowsAdjacent) {
  DataColumn<long double> data(3);
  data << 1.0, 2.0, 3.0;

  std::set<int> indices = { 0, 1 };
  DataColumn<long double> actual = select_rows(data, indices);

  DataColumn<long double> expected(2);
  expected << 1.0, 2.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, SelectGroupIndicesSingleGroup) {
  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    1;

  std::vector<int> actual = select_group(groups, 1);
  std::vector<int> expected = { 0, 1, 2 };

  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, SelectGroupIndicesMultipleGroupsAdjacent) {
  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    2;

  std::vector<int> actual = select_group(groups, 1);
  std::vector<int> expected = { 0, 1 };

  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, SelectGroupIndicesMultipleGroupsMixed) {
  DataColumn<int> groups(3);
  groups <<
    1,
    2,
    1;

  std::vector<int> actual = select_group(groups, 1);
  std::vector<int> expected = { 0, 2 };

  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, SelectGroupIndicesEmptyResult) {
  DataColumn<int> groups(3);
  groups <<
    1,
    1,
    1;

  std::vector<int> actual = select_group(groups, 2);
  std::vector<int> expected = {};

  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, UniqueEmptyResult) {
  DataColumn<int> column(0);
  std::set<int> actual = unique(column);
  std::set<int> expected;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, UniqueSingleValue) {
  DataColumn<int> column(1);
  column << 1;
  std::set<int> actual = unique(column);
  std::set<int> expected = { 1 };

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, UniqueSingleValueRepeated) {
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

TEST(DataColumn, UniqueMultipleValues) {
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

TEST(DataColumn, UniqueMultipleValuesRepeated) {
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

TEST(DataColumn, ExpandIdempotent) {
  DataColumn<long double> data(3);
  data << 1.0, 2.0, 3.0;

  std::vector<int> mask { 1, 1, 1 };

  DataColumn<long double> actual = expand(data, mask);

  ASSERT_EQ(data.size(), actual.size());
  ASSERT_EQ(data.rows(), actual.rows());
  ASSERT_EQ(data.cols(), actual.cols());
  ASSERT_EQ(data, actual);
}

TEST(DataColumn, ExpandGeneric) {
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

TEST(DataColumn, MeanSingleObservation) {
  DataColumn<long double> data(1);
  data <<
    1.0;

  long double actual = mean(data);
  long double expected = 1.0;

  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, MeanMultipleEqualObservations) {
  DataColumn<long double> data(3);
  data <<
    1.0,
    1.0,
    1.0;

  long double actual = mean(data);
  long double expected = 1.0;

  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, MeanMultipleDifferentObservations) {
  DataColumn<long double> data(3);
  data <<
    1.0,
    2.0,
    3.0;

  long double actual = mean(data);
  long double expected = 2.0;

  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, SdZeroVector) {
  DataColumn<long double> data(3);
  data <<
    0,
    0,
    0;

  long double result = sd(data);
  long double expected = 0;

  ASSERT_EQ(expected, result);
}

TEST(DataColumn, SdConstantVector) {
  DataColumn<long double> data(3);
  data <<
    1,
    1,
    1;

  long double result = sd(data);
  long double expected = 0;

  ASSERT_EQ(expected, result);
}

TEST(DataColumn, SdGeneric1) {
  DataColumn<long double> data(3);
  data <<
    1,
    2,
    3;

  long double result = sd(data);
  long double expected = 1;

  ASSERT_EQ(expected, result);
}

TEST(DataColumn, SdGeneric2) {
  DataColumn<long double> data(3);
  data <<
    1,
    1,
    2;

  long double result = sd(data);
  long double expected = 0.5773503;

  ASSERT_NEAR(expected, result, 0.00001);
}

TEST(DataColumn, CenterSingleObservation) {
  DataColumn<long double> data(1);
  data << 1.0;

  DataColumn<long double> actual = center(data);

  DataColumn<long double> expected = DataColumn<long double>::Zero(1);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, CenterMultipleEqualObservations) {
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

TEST(DataColumn, CenterMultipleDifferentObservations) {
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

TEST(DataColumn, DescaleZeroVector) {
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

TEST(DataColumn, DescaleConstantVector) {
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

TEST(DataColumn, DescaleDescaledVector) {
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

TEST(DataColumn, DescaleScaledVector) {
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

TEST(DataColumn, AccuracyMax) {
  DataColumn<long double> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<long double> actual(3);
  actual <<
    1,
    2,
    3;

  double result = accuracy(predictions, actual);
  double expected = 1.0;

  ASSERT_DOUBLE_EQ(expected, result);
}

TEST(DataColumn, AccuracyMin) {
  DataColumn<long double> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<long double> actual(3);
  actual <<
    3,
    3,
    1;

  double result = accuracy(predictions, actual);
  double expected = 0.0;

  ASSERT_DOUBLE_EQ(expected, result);
}

TEST(DataColumn, AccuracyGeneric1) {
  DataColumn<long double> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<long double> actual(3);
  actual <<
    1,
    3,
    3;

  double result = accuracy(predictions, actual);
  double expected = 2.0 / 3.0;

  ASSERT_DOUBLE_EQ(expected, result);
}

TEST(DataColumn, AccuracyGeneric2) {
  DataColumn<long double> predictions(4);
  predictions <<
    1,
    2,
    3,
    4;

  DataColumn<long double> actual(4);
  actual <<
    1,
    1,
    3,
    3;


  double result = accuracy(predictions, actual);
  double expected = 1.0 / 2.0;

  ASSERT_DOUBLE_EQ(expected, result);
}

TEST(DataColumn, AccuracyMorePredictionsThanObservations) {
  DataColumn<int> predictions(3);
  predictions << 0, 1, 2;

  DataColumn<int> observations(2);
  observations << 0, 1;

  ASSERT_THROW(accuracy(predictions, observations), std::invalid_argument);
}

TEST(DataColumn, AccuracyMoreObservationsThanPredictions) {
  DataColumn<int> predictions(2);
  predictions << 0, 1;

  DataColumn<int> observations(3);
  observations << 0, 1, 2;

  ASSERT_THROW(accuracy(predictions, observations), std::invalid_argument);
}

TEST(DataColumn, ErrorRateMax) {
  DataColumn<long double> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<long double> actual(3);
  actual <<
    3,
    3,
    1;

  double result = error_rate(predictions, actual);
  double expected = 1.0;

  ASSERT_DOUBLE_EQ(expected, result);
}

TEST(DataColumn, ErrorRateMin) {
  DataColumn<long double> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<long double> actual(3);
  actual <<
    1,
    2,
    3;

  double result = error_rate(predictions, actual);
  double expected = 0.0;

  ASSERT_DOUBLE_EQ(expected, result);
}

TEST(DataColumn, ErrorRateGeneric1) {
  DataColumn<long double> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<long double> actual(3);
  actual <<
    1,
    3,
    3;

  double result = error_rate(predictions, actual);
  double expected = 1.0 / 3.0;

  ASSERT_DOUBLE_EQ(expected, result);
}

TEST(DataColumn, ErrorRateGeneric2) {
  DataColumn<long double> predictions(4);
  predictions <<
    1,
    2,
    3,
    4;

  DataColumn<long double> actual(4);
  actual <<
    1,
    1,
    3,
    3;

  double result = error_rate(predictions, actual);
  double expected = 1.0 / 2.0;

  ASSERT_DOUBLE_EQ(expected, result);
}

TEST(DataColumn, ErrorRateMorePredictionsThanObservations) {
  DataColumn<int> predictions(3);
  predictions << 0, 1, 2;

  DataColumn<int> observations(2);
  observations << 0, 1;

  ASSERT_THROW(error_rate(predictions, observations), std::invalid_argument);
}

TEST(DataColumn, ErrorRateMoreObservationsThanPredictions) {
  DataColumn<int> predictions(2);
  predictions << 0, 1;

  DataColumn<int> observations(3);
  observations << 0, 1, 2;

  ASSERT_THROW(error_rate(predictions, observations), std::invalid_argument);
}
