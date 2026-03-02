#include <gtest/gtest.h>

#include "stats/Stats.hpp"
#include "utils/Types.hpp"

#include "utils/Macros.hpp"

using namespace pptree;
using namespace pptree::stats;
using namespace pptree::types;

TEST(Stats, Standardize) {
  FeatureMatrix data = DATA(Feature, 3,
      1.0, 3.0, 1.0,
      2.0, 2.0, 3.0,
      3.0, 1.0, 2.0);

  FeatureMatrix standardized = standardize(data);

  FeatureMatrix expected = DATA(Feature, 3,
      -1.0, 1.0, -1.0,
      0.0, 0.0,  1.0,
      1.0, -1.0, 0.0);

  ASSERT_EQ(expected.size(), standardized.size());
  ASSERT_EQ(expected.rows(), standardized.rows());
  ASSERT_EQ(expected.cols(), standardized.cols());
  ASSERT_EQ(expected, standardized);
}

TEST(Stats, Sort) {
  FeatureMatrix x = DATA(Feature, 3,
      1.0, 3.0, 1.0,
      2.0, 2.0, 3.0,
      3.0, 1.0, 2.0);

  ResponseVector y = DATA(Response, 3, 1, 2, 1);

  sort(x, y);

  FeatureMatrix expected_x = DATA(Feature, 3,
      1.0, 3.0, 1.0,
      3.0, 1.0, 2.0,
      2.0, 2.0, 3.0);

  ResponseVector expected_y = DATA(Response, 3, 1, 1, 2);

  ASSERT_EQ_DATA(expected_x, x);
  ASSERT_EQ_DATA(expected_y, y);
}

TEST(Stats, UniqueEmptyResult) {
  ResponseVector column(0);
  std::set<int> actual = unique(column);
  std::set<int> expected;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(Stats, UniqueSingleValue) {
  ResponseVector column  = DATA(Response, 1, 1);
  std::set<int> actual   = unique(column);
  std::set<int> expected = { 1 };

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(Stats, UniqueSingleValueRepeated) {
  ResponseVector column = DATA(Response, 3, 1, 1, 1);

  std::set<int> actual   = unique(column);
  std::set<int> expected = { 1 };

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(Stats, UniqueMultipleValues) {
  ResponseVector column  = DATA(Response, 3, 1, 2, 3);
  std::set<int> actual   = unique(column);
  std::set<int> expected = { 1, 2, 3 };

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(Stats, UniqueMultipleValuesRepeated) {
  ResponseVector column  = DATA(Response, 3, 1, 2, 1);
  std::set<int> actual   = unique(column);
  std::set<int> expected = { 1, 2 };

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(Stats, AccuracyMax) {
  ResponseVector predictions = DATA(Response, 3, 1, 2, 3);
  ResponseVector actual      = DATA(Response, 3, 1, 2, 3);

  float result   = accuracy(predictions, actual);
  float expected = 1.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(Stats, AccuracyMin) {
  ResponseVector predictions = DATA(Response, 3, 1, 2, 3);
  ResponseVector actual      = DATA(Response, 3, 3, 3, 1);

  float result   = accuracy(predictions, actual);
  float expected = 0.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(Stats, AccuracyGeneric1) {
  ResponseVector predictions = DATA(Response, 3, 1, 2, 3);
  ResponseVector actual      = DATA(Response, 3, 1, 3, 3);

  float result   = accuracy(predictions, actual);
  float expected = 2.0 / 3.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(Stats, AccuracyGeneric2) {
  ResponseVector predictions = DATA(Response, 4, 1, 2, 3, 4);
  ResponseVector actual      = DATA(Response, 4, 1, 1, 3, 3);

  float result   = accuracy(predictions, actual);
  float expected = 1.0 / 2.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(Stats, AccuracyMorePredictionsThanObservations) {
  ResponseVector predictions  = DATA(Response, 3, 0, 1, 2);
  ResponseVector observations = DATA(Response, 2, 0, 1);

  ASSERT_THROW(accuracy(predictions, observations), std::invalid_argument);
}

TEST(Stats, AccuracyMoreObservationsThanPredictions) {
  ResponseVector predictions  = DATA(Response, 2, 0, 1);
  ResponseVector observations = DATA(Response, 3, 0, 1, 2);

  ASSERT_THROW(accuracy(predictions, observations), std::invalid_argument);
}

TEST(Stats, ErrorRateMax) {
  ResponseVector predictions = DATA(Response, 3, 1, 2, 3);
  ResponseVector actual      = DATA(Response, 3, 3, 3, 1);

  float result   = error_rate(predictions, actual);
  float expected = 1.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(Stats, ErrorRateMin) {
  ResponseVector predictions = DATA(Response, 3, 1, 2, 3);
  ResponseVector actual      = DATA(Response, 3, 1, 2, 3);

  float result   = error_rate(predictions, actual);
  float expected = 0.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(Stats, ErrorRateGeneric1) {
  ResponseVector predictions = DATA(Response, 3, 1, 2, 3);
  ResponseVector actual      = DATA(Response, 3, 1, 3, 3);

  float result   = error_rate(predictions, actual);
  float expected = 1.0 / 3.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(Stats, ErrorRateGeneric2) {
  ResponseVector predictions = DATA(Response, 4, 1, 2, 3, 4);
  ResponseVector actual      = DATA(Response, 4, 1, 1, 3, 3);

  float result   = error_rate(predictions, actual);
  float expected = 1.0 / 2.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(Stats, ErrorRateMorePredictionsThanObservations) {
  ResponseVector predictions  = DATA(Response, 3, 0, 1, 2);
  ResponseVector observations = DATA(Response, 2, 0, 1);

  ASSERT_THROW(error_rate(predictions, observations), std::invalid_argument);
}

TEST(Stats, ErrorRateMoreObservationsThanPredictions) {
  ResponseVector predictions  = DATA(Response, 2, 0, 1);
  ResponseVector observations = DATA(Response, 3, 0, 1, 2);

  ASSERT_THROW(error_rate(predictions, observations), std::invalid_argument);
}
