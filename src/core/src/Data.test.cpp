#include <gtest/gtest.h>

#include "Data.hpp"

#include "Macros.hpp"

using namespace models::stats;

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

TEST(Data, Standardize) {
  Data<float> data(3, 3);
  data <<
    1.0, 3.0, 1.0,
    2.0, 2.0, 3.0,
    3.0, 1.0, 2.0;

  Data<float> standardized = standardize(data);

  Data<float> expected(3, 3);
  expected <<
    -1.0, 1.0, -1.0,
    0.0, 0.0,  1.0,
    1.0, -1.0, 0.0;

  ASSERT_EQ(expected.size(), standardized.size());
  ASSERT_EQ(expected.rows(), standardized.rows());
  ASSERT_EQ(expected.cols(), standardized.cols());
  ASSERT_EQ(expected, standardized);
}

TEST(Data, Sort) {
  Data<float> x(3, 3);
  x <<
    1.0, 3.0, 1.0,
    2.0, 2.0, 3.0,
    3.0, 1.0, 2.0;

  DataColumn<int> y(3);
  y <<
    1.0,
    2.0,
    1.0;

  sort(x, y);

  Data<float> expected_x(3, 3);
  expected_x <<
    1.0, 3.0, 1.0,
    3.0, 1.0, 2.0,
    2.0, 2.0, 3.0;

  DataColumn<int> expected_y(3);
  expected_y <<
    1.0,
    1.0,
    2.0;

  ASSERT_EQ_MATRIX(expected_x, x);
  ASSERT_EQ_MATRIX(expected_y, y);
}


TEST(Data, StratifiedPorportionalSampleNegativeSampleSize) {
  Random::seed(0);

  Data<float> x(6, 3);
  x <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> y(6);
  y <<
    0,
    0,
    0,
    1,
    1,
    1;

  const std::set<int> classes = { 0, 1 };

  ASSERT_THROW({ stratified_proportional_sample(x, y, classes, -1); }, std::runtime_error);
}

TEST(Data, StratifiedPorportionalSampleZeroSampleSize) {
  Random::seed(0);

  Data<float> x(6, 3);
  x <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> y(6);
  y <<
    0,
    0,
    0,
    1,
    1,
    1;

  const std::set<int> classes = { 0, 1 };

  ASSERT_THROW({ stratified_proportional_sample(x, y, classes, 0); }, std::runtime_error);
}

TEST(Data, StratifiedPorportionalSampleSampleSizeLargerThanRows) {
  Random::seed(0);

  Data<float> x(6, 3);
  x <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> y(6);
  y <<
    0,
    0,
    0,
    1,
    1,
    1;

  const std::set<int> classes = { 0, 1 };

  ASSERT_THROW({ stratified_proportional_sample(x, y, classes, 7); }, std::runtime_error);
}

TEST(Data, StratifiedPorportionalSampleAssertCorrectSize) {
  Random::seed(0);

  Data<float> x(6, 3);
  x <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> y(6);
  y <<
    0,
    0,
    0,
    1,
    1,
    1;

  const std::set<int> classes = { 0, 1 };

  std::vector<int> result = stratified_proportional_sample(x, y, classes, 4);

  ASSERT_EQ(4, result.size());
}

TEST(Data, StratifiedPorportionalSampleAssertCorrectSizePerStrata) {
  Random::seed(0);

  Data<float> x(6, 3);
  x <<
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    3.0, 1.0, 2.0,
    6.0, 4.0, 5.0,
    9.0, 7.0, 8.0;

  DataColumn<int> y(6);
  y <<
    0,
    0,
    0,
    1,
    1,
    1;

  const std::set<int> classes = { 0, 1 };

  std::vector<int> result = stratified_proportional_sample(x, y, classes, 4);

  std::map<int, int> result_sizes;

  for (int i = 0; i < result.size(); i++) {
    int index = result[i];
    int value = y[index];
    result_sizes[value]++;
  }

  ASSERT_EQ(2, result_sizes[0]);
  ASSERT_EQ(2, result_sizes[1]);
}

TEST(Data, StratifiedPorportionalSampleThreeGroupsEqualSize) {
  Random::seed(0);

  Data<float> x(9, 3);

  x <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    4.0, 4.0, 4.0,
    5.0, 5.0, 5.0,
    6.0, 6.0, 6.0,
    7.0, 7.0, 7.0,
    8.0, 8.0, 8.0,
    9.0, 9.0, 9.0;

  DataColumn<int> y(9);
  y <<
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2,
    2;

  const std::set<int> classes = { 0, 1, 2 };

  std::vector<int> result = stratified_proportional_sample(x, y, classes, 6);

  std::map<int, int> result_sizes;

  for (int i = 0; i < result.size(); i++) {
    int index = result[i];
    int value = y[index];
    result_sizes[value]++;
  }

  ASSERT_EQ(6, result.size());
  ASSERT_EQ(2, result_sizes[0]);
  ASSERT_EQ(2, result_sizes[1]);
  ASSERT_EQ(2, result_sizes[2]);
}

TEST(Data, StratifiedPorportionalSampleTwoGroupsDifferentSizeEvenSampleSize) {
  Random::seed(0);

  Data<float> x(9, 3);

  x <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    4.0, 4.0, 4.0,
    5.0, 5.0, 5.0,
    6.0, 6.0, 6.0,
    7.0, 7.0, 7.0,
    8.0, 8.0, 8.0,
    9.0, 9.0, 9.0;

  DataColumn<int> y(9);
  y <<
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1;

  const std::set<int> classes = { 0, 1 };

  std::vector<int> result = stratified_proportional_sample(x, y, classes, 6);

  std::map<int, int> result_sizes;

  for (int i = 0; i < result.size(); i++) {
    int index = result[i];
    int value = y[index];
    result_sizes[value]++;
  }

  ASSERT_EQ(6, result.size());
  ASSERT_EQ(2, result_sizes[0]);
  ASSERT_EQ(4, result_sizes[1]);
}

TEST(Data, StratifiedPorportionalSampleTwoGroupsDifferentSizeOddSampleSize) {
  Random::seed(0);

  Data<float> x(9, 3);

  x <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    4.0, 4.0, 4.0,
    5.0, 5.0, 5.0,
    6.0, 6.0, 6.0,
    7.0, 7.0, 7.0,
    8.0, 8.0, 8.0,
    9.0, 9.0, 9.0;

  DataColumn<int> y(9);
  y <<
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1;

  const std::set<int> classes = { 0, 1 };

  std::vector<int> result = stratified_proportional_sample(x, y, classes, 5);

  std::map<int, int> result_sizes;

  for (int i = 0; i < result.size(); i++) {
    int index = result[i];
    int value = y[index];
    result_sizes[value]++;
  }

  ASSERT_EQ(5, result.size());
  ASSERT_EQ(1, result_sizes[0]);
  ASSERT_EQ(4, result_sizes[1]);
}

TEST(Data, StratifiedPorportionalSampleAtLeastOneObservationPerGroup) {
  Random::seed(0);

  Data<float> x(9, 3);

  x <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    4.0, 4.0, 4.0,
    5.0, 5.0, 5.0,
    6.0, 6.0, 6.0,
    7.0, 7.0, 7.0,
    8.0, 8.0, 8.0,
    9.0, 9.0, 9.0;

  DataColumn<int> y(9);
  y <<
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    2;

  const std::set<int> classes = { 0, 1, 2 };

  std::vector<int> result = stratified_proportional_sample(x, y, classes, 3);

  std::map<int, int> result_sizes;

  for (int i = 0; i < result.size(); i++) {
    int index = result[i];
    int value = y[index];
    result_sizes[value]++;
  }

  ASSERT_EQ(4, result.size());
  ASSERT_EQ(1, result_sizes[0]);
  ASSERT_EQ(2, result_sizes[1]);
  ASSERT_EQ(1, result_sizes[2]);
}
