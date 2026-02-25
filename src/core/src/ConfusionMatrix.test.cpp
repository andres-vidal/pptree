#include <gtest/gtest.h>

#include "ConfusionMatrix.hpp"

#include "Macros.hpp"

using namespace models::stats;

TEST(ConfusionMatrix, Identity) {
  DataColumn<int> actual   = DATA(int, 3, 0, 1, 2);
  DataColumn<int> expected = DATA(int, 3, 0, 1, 2);

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Data<int> expected_result_values = DATA(int, 3,
      1, 0, 0,
      0, 1, 0,
      0, 0, 1);

  ASSERT_EQ(expected_result_values.size(), result.values.size());
  ASSERT_EQ(expected_result_values.rows(), result.values.rows());
  ASSERT_EQ(expected_result_values.cols(), result.values.cols());
  ASSERT_EQ(expected_result_values, result.values);
}

TEST(ConfusionMatrix, Diagonal) {
  DataColumn<int> actual   = DATA(int, 6, 0, 1, 1, 2, 2, 2);
  DataColumn<int> expected = DATA(int, 6, 0, 1, 1, 2, 2, 2);

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Data<int> expected_result_values = DATA(int, 3,
      1, 0, 0,
      0, 2, 0,
      0, 0, 3);

  ASSERT_EQ(expected_result_values.size(), result.values.size());
  ASSERT_EQ(expected_result_values.rows(), result.values.rows());
  ASSERT_EQ(expected_result_values.cols(), result.values.cols());
  ASSERT_EQ(expected_result_values, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}

TEST(ConfusionMatrix, InverseDiagonal) {
  DataColumn<int> actual   = DATA(int, 3, 0, 1, 2);
  DataColumn<int> expected = DATA(int, 3, 2, 1, 0);

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Data<int> expected_result_values = DATA(int, 3,
      0, 0, 1,
      0, 1, 0,
      1, 0, 0);

  ASSERT_EQ(expected_result_values.size(), result.values.size());
  ASSERT_EQ(expected_result_values.rows(), result.values.rows());
  ASSERT_EQ(expected_result_values.cols(), result.values.cols());
  ASSERT_EQ(expected_result_values, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}

TEST(ConfusionMatrix, ZeroDiagonal) {
  DataColumn<int> actual   = DATA(int, 3, 0, 1, 2);
  DataColumn<int> expected = DATA(int, 3, 1, 2, 0);

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Data<int> expected_result_values = DATA(int, 3,
      0, 0, 1,
      1, 0, 0,
      0, 1, 0);

  ASSERT_EQ(expected_result_values.size(), result.values.size());
  ASSERT_EQ(expected_result_values.rows(), result.values.rows());
  ASSERT_EQ(expected_result_values.cols(), result.values.cols());
  ASSERT_EQ(expected_result_values, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}

TEST(ConfusionMatrix, Generic) {
  DataColumn<int> actual   = DATA(int, 6, 0, 1, 1, 2, 2, 2);
  DataColumn<int> expected = DATA(int, 6, 0, 1, 1, 2, 2, 0);

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Data<int> expected_result_values = DATA(int, 3,
      1, 0, 1,
      0, 2, 0,
      0, 0, 2);

  ASSERT_EQ(expected_result_values.size(), result.values.size());
  ASSERT_EQ(expected_result_values.rows(), result.values.rows());
  ASSERT_EQ(expected_result_values.cols(), result.values.cols());
  ASSERT_EQ(expected_result_values, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}

TEST(ConfusionMatrix, MorePredictionsThanObservations) {
  DataColumn<int> predictions  = DATA(int, 3, 0, 1, 2);
  DataColumn<int> observations = DATA(int, 2, 0, 1);

  ASSERT_THROW(ConfusionMatrix(predictions, observations), std::invalid_argument);
}

TEST(ConfusionMatrix, MoreObservationsThanPredictions) {
  DataColumn<int> predictions  = DATA(int, 2, 0, 1);
  DataColumn<int> observations = DATA(int, 3, 0, 1, 2);

  ASSERT_THROW(ConfusionMatrix(predictions, observations), std::invalid_argument);
}

TEST(ConfusionMatrix, ErrorMin) {
  DataColumn<int> actual      = DATA(int, 3, 0, 1, 2);
  DataColumn<int> predictions = DATA(int, 3, 0, 1, 2);

  float result = ConfusionMatrix(predictions, actual).error();

  ASSERT_FLOAT_EQ(0, result);
}

TEST(ConfusionMatrix, ErrorMax) {
  DataColumn<int> actual      = DATA(int, 3, 0, 1, 2);
  DataColumn<int> predictions = DATA(int, 3, 2, 0, 1);

  float result = ConfusionMatrix(predictions, actual).error();

  ASSERT_FLOAT_EQ(1, result);
}

TEST(ConfusionMatrix, ErrorGeneric) {
  DataColumn<int> actual      = DATA(int, 6, 0, 1, 1, 2, 2, 2);
  DataColumn<int> predictions = DATA(int, 6, 0, 1, 2, 0, 1, 2);

  float result = ConfusionMatrix(predictions, actual).error();

  ASSERT_NEAR(0.5, result, 0.0001);
}

TEST(ConfusionMatrix, ClassErrorsAllZero) {
  DataColumn<int> actual      = DATA(int, 3, 0, 1, 2);
  DataColumn<int> predictions = DATA(int, 3, 0, 1, 2);

  DataColumn<float> result = ConfusionMatrix(predictions, actual).class_errors();

  DataColumn<float> expected_errors = DATA(float, 3, 0, 0, 0);

  ASSERT_EQ(expected_errors.size(), result.size());
  ASSERT_EQ(expected_errors, result);
}

TEST(ConfusionMatrix, ClassErrorsAllOne) {
  DataColumn<int> actual      = DATA(int, 6, 0, 1, 1, 2, 2, 2);
  DataColumn<int> predictions = DATA(int, 6, 1, 2, 2, 1, 1, 1);

  DataColumn<float> result = ConfusionMatrix(predictions, actual).class_errors();

  DataColumn<float> expected_errors = DATA(float, 3, 1, 1, 1);

  ASSERT_EQ(expected_errors.size(), result.size());
  ASSERT_EQ(expected_errors, result);
}

TEST(ConfusionMatrix, ClassErrorsMixed) {
  DataColumn<int> actual      = DATA(int, 6, 0, 1, 1, 2, 2, 2);
  DataColumn<int> predictions = DATA(int, 6, 0, 1, 2, 0, 1, 2);

  DataColumn<float> result = ConfusionMatrix(predictions, actual).class_errors();

  DataColumn<float> expected_errors = DATA(float, 3, 0, 0.5, 0.666667);

  ASSERT_EQ(expected_errors.size(), result.size());
  ASSERT_APPROX(expected_errors, result);
}
