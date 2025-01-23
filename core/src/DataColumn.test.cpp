#include <gtest/gtest.h>

#include "DataColumn.hpp"

using namespace models::stats;

TEST(DataColumn, SelectRowsVectorSingleRow) {
  DataColumn<double> data(3);
  data << 1.0, 2.0, 3.0;

  std::vector<int> indices = { 1 };
  DataColumn<double> actual = select_rows(data, indices);

  DataColumn<double> expected(1);
  expected << 2.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, SelectRowsVectorMultipleRowsNonAdjacent) {
  DataColumn<double> data(3);
  data << 1.0, 2.0, 3.0;

  std::vector<int> indices = { 0, 2 };
  DataColumn<double> actual = select_rows(data, indices);

  DataColumn<double> expected(2);
  expected << 1.0, 3.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, SelectRowsVectorMultipleRowsAdjacent) {
  DataColumn<double> data(3);
  data << 1.0, 2.0, 3.0;

  std::vector<int> indices = { 0, 1 };
  DataColumn<double> actual = select_rows(data, indices);

  DataColumn<double> expected(2);
  expected << 1.0, 2.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, SelectRowsSetSingleRow) {
  DataColumn<double> data(3);
  data << 1.0, 2.0, 3.0;

  std::set<int> indices = { 1 };
  DataColumn<double> actual = select_rows(data, indices);

  DataColumn<double> expected(1);
  expected << 2.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, SelectRowsSetMultipleRowsNonAdjacent) {
  DataColumn<double> data(3);
  data << 1.0, 2.0, 3.0;

  std::set<int> indices = { 0, 2 };
  DataColumn<double> actual = select_rows(data, indices);

  DataColumn<double> expected(2);
  expected << 1.0, 3.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, SelectRowsSetMultipleRowsAdjacent) {
  DataColumn<double> data(3);
  data << 1.0, 2.0, 3.0;

  std::set<int> indices = { 0, 1 };
  DataColumn<double> actual = select_rows(data, indices);

  DataColumn<double> expected(2);
  expected << 1.0, 2.0;

  ASSERT_EQ(expected.size(), actual.size());
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

TEST(DataColumn, SdZeroVector) {
  DataColumn<double> data(3);
  data <<
    0,
    0,
    0;

  double result = sd(data);
  double expected = 0;

  ASSERT_EQ(expected, result);
}

TEST(DataColumn, SdConstantVector) {
  DataColumn<double> data(3);
  data <<
    1,
    1,
    1;

  double result = sd(data);
  double expected = 0;

  ASSERT_EQ(expected, result);
}

TEST(DataColumn, SdGeneric1) {
  DataColumn<double> data(3);
  data <<
    1,
    2,
    3;

  double result = sd(data);
  double expected = 1;

  ASSERT_EQ(expected, result);
}

TEST(DataColumn, SdGeneric2) {
  DataColumn<double> data(3);
  data <<
    1,
    1,
    2;

  double result = sd(data);
  double expected = 0.5773503;

  ASSERT_NEAR(expected, result, 0.00001);
}

TEST(DataColumn, CenterSingleObservation) {
  DataColumn<double> data(1);
  data << 1.0;

  DataColumn<double> actual = center(data);

  DataColumn<double> expected = DataColumn<double>::Zero(1);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, CenterMultipleEqualObservations) {
  DataColumn<double> data(3);
  data <<
    1.0,
    1.0,
    1.0;

  DataColumn<double> actual = center(data);

  DataColumn<double> expected = DataColumn<double>::Zero(3);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, CenterMultipleDifferentObservations) {
  DataColumn<double> data(3);
  data <<
    1.0,
    2.0,
    3.0;

  DataColumn<double> actual = center(data);

  DataColumn<double> expected(3);
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
  DataColumn<double> data(3);
  data <<
    0,
    0,
    0;

  DataColumn<double> actual = descale(data);

  DataColumn<double> expected(3);
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
  DataColumn<double> data(3);
  data <<
    1,
    1,
    1;

  DataColumn<double> actual = descale(data);

  DataColumn<double> expected(3);
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
  DataColumn<double> data(3);
  data <<
    1,
    2,
    3;

  DataColumn<double> actual = descale(data);

  DataColumn<double> expected(3);
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
  DataColumn<double> data(3);
  data <<
    2,
    4,
    6;

  DataColumn<double> actual = descale(data);

  DataColumn<double> expected(3);
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
  DataColumn<double> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<double> actual(3);
  actual <<
    1,
    2,
    3;

  double result = accuracy(predictions, actual);
  double expected = 1.0;

  ASSERT_DOUBLE_EQ(expected, result);
}

TEST(DataColumn, AccuracyMin) {
  DataColumn<double> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<double> actual(3);
  actual <<
    3,
    3,
    1;

  double result = accuracy(predictions, actual);
  double expected = 0.0;

  ASSERT_DOUBLE_EQ(expected, result);
}

TEST(DataColumn, AccuracyGeneric1) {
  DataColumn<double> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<double> actual(3);
  actual <<
    1,
    3,
    3;

  double result = accuracy(predictions, actual);
  double expected = 2.0 / 3.0;

  ASSERT_DOUBLE_EQ(expected, result);
}

TEST(DataColumn, AccuracyGeneric2) {
  DataColumn<double> predictions(4);
  predictions <<
    1,
    2,
    3,
    4;

  DataColumn<double> actual(4);
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
  DataColumn<double> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<double> actual(3);
  actual <<
    3,
    3,
    1;

  double result = error_rate(predictions, actual);
  double expected = 1.0;

  ASSERT_DOUBLE_EQ(expected, result);
}

TEST(DataColumn, ErrorRateMin) {
  DataColumn<double> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<double> actual(3);
  actual <<
    1,
    2,
    3;

  double result = error_rate(predictions, actual);
  double expected = 0.0;

  ASSERT_DOUBLE_EQ(expected, result);
}

TEST(DataColumn, ErrorRateGeneric1) {
  DataColumn<double> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<double> actual(3);
  actual <<
    1,
    3,
    3;

  double result = error_rate(predictions, actual);
  double expected = 1.0 / 3.0;

  ASSERT_DOUBLE_EQ(expected, result);
}

TEST(DataColumn, ErrorRateGeneric2) {
  DataColumn<double> predictions(4);
  predictions <<
    1,
    2,
    3,
    4;

  DataColumn<double> actual(4);
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
