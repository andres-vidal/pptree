#include <gtest/gtest.h>

#include "ConfusionMatrix.hpp"

using namespace models::stats;

TEST(ConfusionMatrix, Identity) {
  DataColumn<int> actual(3);
  actual << 0, 1, 2;

  DataColumn<int> expected(3);
  expected << 0, 1, 2;

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Data<int> expected_result_values(3, 3);
  expected_result_values <<
    1, 0, 0,
    0, 1, 0,
    0, 0, 1;

  ASSERT_EQ(expected_result_values.size(), result.values.size());
  ASSERT_EQ(expected_result_values.rows(), result.values.rows());
  ASSERT_EQ(expected_result_values.cols(), result.values.cols());
  ASSERT_EQ(expected_result_values, result.values);
}

TEST(ConfusionMatrix, Diagonal) {
  DataColumn<int> actual(6);
  actual << 0, 1, 1, 2, 2, 2;

  DataColumn<int> expected(6);
  expected << 0, 1, 1, 2, 2, 2;

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Data<int> expected_result_values(3, 3);
  expected_result_values <<
    1, 0, 0,
    0, 2, 0,
    0, 0, 3;

  ASSERT_EQ(expected_result_values.size(), result.values.size());
  ASSERT_EQ(expected_result_values.rows(), result.values.rows());
  ASSERT_EQ(expected_result_values.cols(), result.values.cols());
  ASSERT_EQ(expected_result_values, result.values);

  ASSERT_EQ(std::set<int>({ 0, 1, 2 }), result.labels);
}

TEST(ConfusionMatrix, InverseDiagonal) {
  DataColumn<int> actual(3);
  actual << 0, 1, 2;

  DataColumn<int> expected(3);
  expected << 2, 1, 0;

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Data<int> expected_result_values(3, 3);
  expected_result_values <<
    0, 0, 1,
    0, 1, 0,
    1, 0, 0;

  ASSERT_EQ(expected_result_values.size(), result.values.size());
  ASSERT_EQ(expected_result_values.rows(), result.values.rows());
  ASSERT_EQ(expected_result_values.cols(), result.values.cols());
  ASSERT_EQ(expected_result_values, result.values);

  ASSERT_EQ(std::set<int>({ 0, 1, 2 }), result.labels);
}

TEST(ConfusionMatrix, ZeroDiagonal) {
  DataColumn<int> actual(3);
  actual << 0, 1, 2;

  DataColumn<int> expected(3);
  expected << 1, 2, 0;

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Data<int> expected_result_values(3, 3);
  expected_result_values <<
    0, 0, 1,
    1, 0, 0,
    0, 1, 0;

  ASSERT_EQ(expected_result_values.size(), result.values.size());
  ASSERT_EQ(expected_result_values.rows(), result.values.rows());
  ASSERT_EQ(expected_result_values.cols(), result.values.cols());
  ASSERT_EQ(expected_result_values, result.values);

  ASSERT_EQ(std::set<int>({ 0, 1, 2 }), result.labels);
}

TEST(ConfusionMatrix, Generic) {
  DataColumn<int> actual(6);
  actual << 0, 1, 1, 2, 2, 2;

  DataColumn<int> expected(6);
  expected << 0, 1, 1, 2, 2, 0;

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Data<int> expected_result_values(3, 3);
  expected_result_values <<
    1, 0, 1,
    0, 2, 0,
    0, 0, 2;

  ASSERT_EQ(expected_result_values.size(), result.values.size());
  ASSERT_EQ(expected_result_values.rows(), result.values.rows());
  ASSERT_EQ(expected_result_values.cols(), result.values.cols());
  ASSERT_EQ(expected_result_values, result.values);

  ASSERT_EQ(std::set<int>({ 0, 1, 2 }), result.labels);
}

TEST(ConfusionMatrix, MorePredictionsThanObservations) {
  DataColumn<int> predictions(3);
  predictions << 0, 1, 2;

  DataColumn<int> observations(2);
  observations << 0, 1;

  ASSERT_THROW(ConfusionMatrix(predictions, observations), std::invalid_argument);
}

TEST(ConfusionMatrix, MoreObservationsThanPredictions) {
  DataColumn<int> predictions(2);
  predictions << 0, 1;

  DataColumn<int> observations(3);
  observations << 0, 1, 2;

  ASSERT_THROW(ConfusionMatrix(predictions, observations), std::invalid_argument);
}
