#include <gtest/gtest.h>

#include "DataColumn.hpp"

using namespace models::stats;



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
  std::set<int> actual   = unique(column);
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

  std::set<int> actual   = unique(column);
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
  std::set<int> actual   = unique(column);
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
  std::set<int> actual   = unique(column);
  std::set<int> expected = { 1, 2 };

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(DataColumn, AccuracyMax) {
  DataColumn<float> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<float> actual(3);
  actual <<
    1,
    2,
    3;

  float result   = accuracy(predictions, actual);
  float expected = 1.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(DataColumn, AccuracyMin) {
  DataColumn<float> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<float> actual(3);
  actual <<
    3,
    3,
    1;

  float result   = accuracy(predictions, actual);
  float expected = 0.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(DataColumn, AccuracyGeneric1) {
  DataColumn<float> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<float> actual(3);
  actual <<
    1,
    3,
    3;

  float result   = accuracy(predictions, actual);
  float expected = 2.0 / 3.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(DataColumn, AccuracyGeneric2) {
  DataColumn<float> predictions(4);
  predictions <<
    1,
    2,
    3,
    4;

  DataColumn<float> actual(4);
  actual <<
    1,
    1,
    3,
    3;


  float result   = accuracy(predictions, actual);
  float expected = 1.0 / 2.0;

  ASSERT_FLOAT_EQ(expected, result);
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
  DataColumn<float> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<float> actual(3);
  actual <<
    3,
    3,
    1;

  float result   = error_rate(predictions, actual);
  float expected = 1.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(DataColumn, ErrorRateMin) {
  DataColumn<float> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<float> actual(3);
  actual <<
    1,
    2,
    3;

  float result   = error_rate(predictions, actual);
  float expected = 0.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(DataColumn, ErrorRateGeneric1) {
  DataColumn<float> predictions(3);
  predictions <<
    1,
    2,
    3;

  DataColumn<float> actual(3);
  actual <<
    1,
    3,
    3;

  float result   = error_rate(predictions, actual);
  float expected = 1.0 / 3.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(DataColumn, ErrorRateGeneric2) {
  DataColumn<float> predictions(4);
  predictions <<
    1,
    2,
    3,
    4;

  DataColumn<float> actual(4);
  actual <<
    1,
    1,
    3,
    3;

  float result   = error_rate(predictions, actual);
  float expected = 1.0 / 2.0;

  ASSERT_FLOAT_EQ(expected, result);
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
