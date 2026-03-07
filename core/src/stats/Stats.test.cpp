#include <gtest/gtest.h>

#include "stats/Stats.hpp"
#include "utils/Types.hpp"

#include "utils/Macros.hpp"

using namespace pptree;
using namespace pptree::stats;
using namespace pptree::types;

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

TEST(Stats, SortAlreadySorted) {
  FeatureMatrix x = DATA(Feature, 3,
      1.0, 10.0,
      2.0, 20.0,
      3.0, 30.0);

  ResponseVector y = DATA(Response, 3, 0, 0, 1);

  sort(x, y);

  ResponseVector expected_y = DATA(Response, 3, 0, 0, 1);

  ASSERT_EQ(expected_y, y);
}

TEST(Stats, SortReverseSorted) {
  FeatureMatrix x = DATA(Feature, 4,
      4.0,
      3.0,
      2.0,
      1.0);

  ResponseVector y = DATA(Response, 4, 2, 1, 1, 0);

  sort(x, y);

  ResponseVector expected_y = DATA(Response, 4, 0, 1, 1, 2);

  ASSERT_EQ(expected_y, y);
  // Feature matrix should be reordered to match
  ASSERT_FLOAT_EQ(1.0, x(0, 0));
  ASSERT_FLOAT_EQ(4.0, x(3, 0));
}

TEST(Stats, SortLargerArray) {
  FeatureMatrix x = DATA(Feature, 6,
      6.0, 5.0, 4.0,
      3.0, 2.0, 1.0);

  ResponseVector y = DATA(Response, 6, 2, 1, 0, 2, 1, 0);

  sort(x, y);

  for (int i = 1; i < y.size(); ++i) {
    ASSERT_LE(y[i - 1], y[i]);
  }
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

TEST(Stats, SdVector) {
  FeatureVector v = DATA(Feature, 4, 2.0f, 4.0f, 4.0f, 4.0f);
  // mean = 3.5, var = ((−1.5)²+(0.5)²+(0.5)²+(0.5)²) / 3 = 3/3 = 1, sd = 1
  ASSERT_NEAR(sd(v), 1.0, 1e-5);
}

TEST(Stats, SdVectorSingleElement) {
  FeatureVector v = DATA(Feature, 1, 5.0f);
  ASSERT_DOUBLE_EQ(sd(v), 0.0);
}

TEST(Stats, SdMatrixSingleColumn) {
  FeatureMatrix m = DATA(Feature, 4,
      2.0f,
      4.0f,
      4.0f,
      4.0f);

  FeatureVector result = sd(m);

  ASSERT_EQ(result.size(), 1);
  ASSERT_NEAR(result(0), 1.0f, 1e-5f);
}

TEST(Stats, SdMatrixMultipleColumns) {
  // col 0: {1, 2, 3} → mean=2, var=1, sd=1
  // col 1: {4, 4, 4} → mean=4, var=0, sd=0
  // col 2: {0, 5, 10} → mean=5, var=25, sd=5
  FeatureMatrix m = DATA(Feature, 3,
      1.0f, 4.0f,  0.0f,
      2.0f, 4.0f,  5.0f,
      3.0f, 4.0f, 10.0f);

  FeatureVector result = sd(m);

  ASSERT_EQ(result.size(), 3);
  ASSERT_NEAR(result(0), 1.0f, 1e-5f);
  ASSERT_NEAR(result(1), 0.0f, 1e-5f);
  ASSERT_NEAR(result(2), 5.0f, 1e-5f);
}

TEST(Stats, SdMatrixMatchesVectorSd) {
  FeatureMatrix m = DATA(Feature, 5,
      1.0f, 10.0f,
      3.0f, 20.0f,
      5.0f, 30.0f,
      7.0f, 40.0f,
      9.0f, 50.0f);

  FeatureVector result = sd(m);

  ASSERT_NEAR(result(0), static_cast<float>(sd(static_cast<FeatureVector>(m.col(0)))), 1e-5f);
  ASSERT_NEAR(result(1), static_cast<float>(sd(static_cast<FeatureVector>(m.col(1)))), 1e-5f);
}
