#include <gtest/gtest.h>

#include "ConfusionMatrix.hpp"
#include "Types.hpp"

#include "Macros.hpp"

using namespace models;
using namespace models::stats;
using namespace models::types;

TEST(ConfusionMatrix, Identity) {
  ResponseVector actual   = DATA(Response, 3, 0, 1, 2);
  ResponseVector expected = DATA(Response, 3, 0, 1, 2);

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Matrix<int> expected_result_values = DATA(Response, 3,
      1, 0, 0,
      0, 1, 0,
      0, 0, 1);

  ASSERT_EQ(expected_result_values.size(), result.values.size());
  ASSERT_EQ(expected_result_values.rows(), result.values.rows());
  ASSERT_EQ(expected_result_values.cols(), result.values.cols());
  ASSERT_EQ(expected_result_values, result.values);
}

TEST(ConfusionMatrix, Diagonal) {
  ResponseVector actual   = DATA(Response, 6, 0, 1, 1, 2, 2, 2);
  ResponseVector expected = DATA(Response, 6, 0, 1, 1, 2, 2, 2);

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Matrix<int> expected_result_values = DATA(Response, 3,
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
  ResponseVector actual   = DATA(Response, 3, 0, 1, 2);
  ResponseVector expected = DATA(Response, 3, 2, 1, 0);

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Matrix<int> expected_result_values = DATA(Response, 3,
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
  ResponseVector actual   = DATA(Response, 3, 0, 1, 2);
  ResponseVector expected = DATA(Response, 3, 1, 2, 0);

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Matrix<int> expected_result_values = DATA(Response, 3,
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
  ResponseVector actual   = DATA(Response, 6, 0, 1, 1, 2, 2, 2);
  ResponseVector expected = DATA(Response, 6, 0, 1, 1, 2, 2, 0);

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Matrix<int> expected_result_values = DATA(Response, 3,
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
  ResponseVector predictions  = DATA(Response, 3, 0, 1, 2);
  ResponseVector observations = DATA(Response, 2, 0, 1);

  ASSERT_THROW(ConfusionMatrix(predictions, observations), std::invalid_argument);
}

TEST(ConfusionMatrix, MoreObservationsThanPredictions) {
  ResponseVector predictions  = DATA(Response, 2, 0, 1);
  ResponseVector observations = DATA(Response, 3, 0, 1, 2);

  ASSERT_THROW(ConfusionMatrix(predictions, observations), std::invalid_argument);
}

TEST(ConfusionMatrix, ErrorMin) {
  ResponseVector actual      = DATA(Response, 3, 0, 1, 2);
  ResponseVector predictions = DATA(Response, 3, 0, 1, 2);

  float result = ConfusionMatrix(predictions, actual).error();

  ASSERT_FLOAT_EQ(0, result);
}

TEST(ConfusionMatrix, ErrorMax) {
  ResponseVector actual      = DATA(Response, 3, 0, 1, 2);
  ResponseVector predictions = DATA(Response, 3, 2, 0, 1);

  float result = ConfusionMatrix(predictions, actual).error();

  ASSERT_FLOAT_EQ(1, result);
}

TEST(ConfusionMatrix, ErrorGeneric) {
  ResponseVector actual      = DATA(Response, 6, 0, 1, 1, 2, 2, 2);
  ResponseVector predictions = DATA(Response, 6, 0, 1, 2, 0, 1, 2);

  float result = ConfusionMatrix(predictions, actual).error();

  ASSERT_NEAR(0.5, result, 0.0001);
}

TEST(ConfusionMatrix, ClassErrorsAllZero) {
  ResponseVector actual      = DATA(Response, 3, 0, 1, 2);
  ResponseVector predictions = DATA(Response, 3, 0, 1, 2);

  FeatureVector result = ConfusionMatrix(predictions, actual).class_errors();

  FeatureVector expected_errors = DATA(Feature, 3, 0, 0, 0);

  ASSERT_EQ(expected_errors.size(), result.size());
  ASSERT_EQ(expected_errors, result);
}

TEST(ConfusionMatrix, ClassErrorsAllOne) {
  ResponseVector actual      = DATA(Response, 6, 0, 1, 1, 2, 2, 2);
  ResponseVector predictions = DATA(Response, 6, 1, 2, 2, 1, 1, 1);

  FeatureVector result = ConfusionMatrix(predictions, actual).class_errors();

  FeatureVector expected_errors = DATA(Feature, 3, 1, 1, 1);

  ASSERT_EQ(expected_errors.size(), result.size());
  ASSERT_EQ(expected_errors, result);
}

TEST(ConfusionMatrix, ClassErrorsMixed) {
  ResponseVector actual      = DATA(Response, 6, 0, 1, 1, 2, 2, 2);
  ResponseVector predictions = DATA(Response, 6, 0, 1, 2, 0, 1, 2);

  FeatureVector result = ConfusionMatrix(predictions, actual).class_errors();

  FeatureVector expected_errors = DATA(Feature, 3, 0, 0.5, 0.666667);

  ASSERT_EQ(expected_errors.size(), result.size());
  ASSERT_APPROX(expected_errors, result);
}
