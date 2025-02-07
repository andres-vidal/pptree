#include <gtest/gtest.h>

#include "Data.hpp"

#include "Macros.hpp"

using namespace models::stats;

TEST(Data, SelectRowsVectorSingleRow) {
  Data<float> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::vector<int> indices = { 1 };
  Data<float> actual = select_rows(data, indices);

  Data<float> expected(1, 3);
  expected <<
    4.0, 5.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, SelectRowsVectorMultipleRowsNonAdjacent) {
  Data<float> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::vector<int> indices = { 0, 2 };
  Data<float> actual = select_rows(data, indices);

  Data<float> expected(2, 3);
  expected <<
    1.0, 2.0, 3.0,
    7.0, 8.0, 9.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, SelectRowsVectorMultipleRowsAdjacent) {
  Data<float> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::vector<int> indices = { 0, 1 };
  Data<float> actual = select_rows(data, indices);

  Data<float> expected(2, 3);
  expected <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, SelectRowsSetSingleRow) {
  Data<float> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::set<int> indices = { 1 };
  Data<float> actual = select_rows(data, indices);

  Data<float> expected(1, 3);
  expected <<
    4.0, 5.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, SelectRowsSetMultipleRowsNonAdjacent) {
  Data<float> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::set<int> indices = { 0, 2 };
  Data<float> actual = select_rows(data, indices);

  Data<float> expected(2, 3);
  expected <<
    1.0, 2.0, 3.0,
    7.0, 8.0, 9.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, SelectRowsSetMultipleRowsAdjacent) {
  Data<float> data(3, 3);
  data <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0;

  std::set<int> indices = { 0, 1 };
  Data<float> actual = select_rows(data, indices);

  Data<float> expected(2, 3);
  expected <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, MeanSingleObservation) {
  Data<float> data(1, 3);
  data <<
    1.0, 2.0, 6.0;

  DataColumn<float> actual = mean(data);

  DataColumn<float> expected(3);
  expected <<
    1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, MeanMultipleEqualObservations) {
  Data<float> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0;

  DataColumn<float> actual = mean(data);

  DataColumn<float> expected(3);
  expected <<
    1.0, 2.0, 6.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, MeanMultipleDifferentObservations) {
  Data<float> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<float> actual = mean(data);

  DataColumn<float> expected(3);
  expected <<
    2.0, 3.0, 7.0;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, CovarianceZeroMatrix) {
  Data<float> data(3, 3);
  data <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  Data<float> result = covariance(data);

  ASSERT_EQ(data.size(), result.size());
  ASSERT_EQ(data.rows(), result.rows());
  ASSERT_EQ(data.cols(), result.cols());
  ASSERT_EQ(data, result);
}

TEST(Data, CovarianceConstantMatrix) {
  Data<float> data(3, 3);
  data <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  Data<float> result = covariance(data);

  Data<float> expected(3, 3);
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
  Data<float> data(3, 3);
  data <<
    1, 2, 3,
    1, 2, 3,
    1, 2, 3;

  Data<float> result = covariance(data);

  Data<float> expected(3, 3);
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
  Data<float> data(3, 3);
  data <<
    1, 2, 3,
    1, 3, 3,
    1, 4, 3;

  Data<float> result = covariance(data);

  Data<float> expected(3, 3);
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
  Data<float> data(3, 3);
  data <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  Data<float> result = covariance(data);

  Data<float> expected(3, 3);
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
  Data<float> data(3, 3);
  data <<
    1, 2, 3,
    1, 2, 3,
    2, 3, 4;

  Data<float> result = covariance(data);

  Data<float> expected(3, 3);
  expected <<
    0.33333, 0.33333, 0.33333,
    0.33333, 0.33333, 0.33333,
    0.33333, 0.33333, 0.33333;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_APPROX(expected, result);
}

TEST(Data, SdZeroMatrix) {
  Data<float> data(3, 3);
  data <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  DataColumn<float> result = sd(data);

  DataColumn<float> expected = DataColumn<float>::Zero(3);

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(Data, SdConstantMatrix) {
  Data<float> data(3, 3);
  data <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  Data<float> result = sd(data);

  DataColumn<float> expected = DataColumn<float>::Zero(3);

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(Data, SdAllConstantColumns) {
  Data<float> data(3, 3);
  data <<
    1, 2, 3,
    1, 2, 3,
    1, 2, 3;

  Data<float> result = sd(data);

  DataColumn<float> expected = DataColumn<float>::Zero(3);

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(Data, SdSomeConstantColumns) {
  Data<float> data(3, 3);
  data <<
    1, 2, 3,
    1, 3, 3,
    1, 4, 3;

  Data<float> result = sd(data);

  DataColumn<float> expected(3);
  expected << 0, 1, 0;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(Data, SdGeneric1) {
  Data<float> data(3, 3);
  data <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  Data<float> result = sd(data);

  DataColumn<float> expected(3);
  expected << 1, 1, 1;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_EQ(expected, result);
}

TEST(Data, SdGeneric2) {
  Data<float> data(3, 3);
  data <<
    1, 2, 3,
    1, 2, 3,
    2, 3, 4;

  Data<float> result = sd(data);

  DataColumn<float> expected(3);
  expected << 0.5773503, 0.5773503, 0.5773503;

  ASSERT_EQ(expected.size(), result.size());
  ASSERT_EQ(expected.rows(), result.rows());
  ASSERT_EQ(expected.cols(), result.cols());
  ASSERT_APPROX(expected, result);
}

TEST(Data, CenterDataSingleObservation) {
  Data<float> data(1, 3);
  data <<
    1.0, 2.0, 6.0;

  Data<float> actual = center(data);

  Data<float> expected = Data<float>::Zero(1, 3);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, CenterDataMultipleEqualObservations) {
  Data<float> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0;

  Data<float> actual = center(data);

  Data<float> expected = Data<float>::Zero(3, 3);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(Data, CenterDataMultipleDifferentObservations) {
  Data<float> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  Data<float> actual = center(data);

  Data<float> expected(3, 3);
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
  Data<float> data(3, 3);
  data <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  Data<float> actual = descale(data);

  Data<float> expected(3, 3);
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
  Data<float> data(3, 3);
  data <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  Data<float> actual = descale(data);

  Data<float> expected(3, 3);
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
  Data<float> data(3, 3);
  data <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  Data<float> actual = descale(data);

  Data<float> expected(3, 3);
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
  Data<float> data(3, 3);
  data <<
    2, 4, 6,
    4, 6, 8,
    6, 8, 10;

  Data<float> actual = descale(data);

  Data<float> expected(3, 3);
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
  Data<float> data(3, 3);
  data <<
    2, 4, 3,
    4, 6, 4,
    6, 8, 5;

  Data<float> actual = descale(data);

  Data<float> expected(3, 3);
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

  Data<float> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  Data<float> shuffled = shuffle_column(data, 0);

  Data<float> expected(3, 3);
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

  Data<float> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  Data<float> shuffled = shuffle_column(data, 1);

  Data<float> expected(3, 3);
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

  Data<float> data(3, 3);
  data <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  Data<float> shuffled = shuffle_column(data, 2);

  Data<float> expected(3, 3);
  expected <<
    1.0, 2.0, 7.0,
    2.0, 3.0, 6.0,
    3.0, 4.0, 8.0;

  ASSERT_EQ(expected.size(), shuffled.size());
  ASSERT_EQ(expected.rows(), shuffled.rows());
  ASSERT_EQ(expected.cols(), shuffled.cols());
  ASSERT_EQ(expected, shuffled);
}
