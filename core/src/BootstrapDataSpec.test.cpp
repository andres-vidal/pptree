#include <gtest/gtest.h>

#include "BootstrapDataSpec.hpp"

using namespace models::stats;

TEST(StratifiedProportionalSample, NegativeSampleSize) {
  Random::seed(0);

  Data<long double> x(6, 3);
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

  DataSpec<long double, int> data(x, y, { 0, 1 });

  ASSERT_THROW({ stratified_proportional_sample(data, -1); }, std::runtime_error);
}

TEST(StratifiedProportionalSample, ZeroSampleSize) {
  Random::seed(0);

  Data<long double> x(6, 3);
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

  DataSpec<long double, int> data(x, y, { 0, 1 });

  ASSERT_THROW({ stratified_proportional_sample(data, 0); }, std::runtime_error);
}

TEST(StratifiedProportionalSample, SampleSizeLargerThanRows) {
  Random::seed(0);

  Data<long double> x(6, 3);
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

  DataSpec<long double, int> data(x, y, { 0, 1 });

  ASSERT_THROW({ stratified_proportional_sample(data, 7); }, std::runtime_error);
}

TEST(StratifiedProportionalSample, AssertCorrectSize) {
  Random::seed(0);

  Data<long double> x(6, 3);
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

  DataSpec<long double, int> data(x, y, { 0, 1 });

  DataSpec<long double, int> result = stratified_proportional_sample(data, 4).get_sample();

  ASSERT_EQ(4, result.x.rows());
  ASSERT_EQ(3, result.x.cols());
  ASSERT_EQ(4, result.y.size());
}

TEST(StratifiedProportionalSample, AssertCorrectSizePerStrata) {
  Random::seed(0);

  Data<long double> x(6, 3);
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

  DataSpec<long double, int> data(x, y, { 0, 1 });

  DataSpec<long double, int> result = stratified_proportional_sample(data, 4).get_sample();

  std::map<int, int> result_sizes;

  for (int i = 0; i < result.y.size(); i++) {
    result_sizes[result.y[i]]++;
  }

  ASSERT_EQ(2, result_sizes[0]);
  ASSERT_EQ(2, result_sizes[1]);
}

TEST(StratifiedProportionalSample, AssertSubsetOfDataPerStrata) {
  Random::seed(0);

  Data<long double> x(6, 3);
  x <<
    1.0, 1.0, 1.0,
    2.0, 2.0, 2.0,
    3.0, 3.0, 3.0,
    4.0, 4.0, 4.0,
    5.0, 5.0, 5.0,
    6.0, 6.0, 6.0;

  DataColumn<int> y(6);
  y <<
    0,
    0,
    0,
    1,
    1,
    1;

  DataSpec<long double, int> data(x, y, { 0, 1 });

  DataSpec<long double, int> result = stratified_proportional_sample(data, 4).get_sample();

  for (int i = 0; i < result.x.rows(); i++) {
    bool found = false;

    for (int j = 0; j < data.x.rows(); j++) {
      if (result.x.row(i) == data.x.row(j) && result.y[i] == data.y[j]) {
        found = true;
        break;
      }
    }

    ASSERT_TRUE(found) << "Expected to find row [" << result.x.row(i) << "] in the original data: " << std::endl << data.x << std::endl;
  }
}

TEST(StratifiedProportionalSample, ThreeGroupsEqualSize) {
  Random::seed(0);

  Data<long double> x(9, 3);

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

  DataSpec<long double, int> data(x, y, { 0, 1, 2 });

  DataSpec<long double, int> result = stratified_proportional_sample(data, 6).get_sample();

  std::map<int, int> result_sizes;

  for (int i = 0; i < result.y.size(); i++) {
    result_sizes[result.y[i]]++;
  }

  ASSERT_EQ(6, result.x.rows());
  ASSERT_EQ(2, result_sizes[0]);
  ASSERT_EQ(2, result_sizes[1]);
  ASSERT_EQ(2, result_sizes[2]);

  for (int i = 0; i < result.x.rows(); i++) {
    bool found = false;

    for (int j = 0; j < data.x.rows(); j++) {
      if (result.x.row(i) == data.x.row(j) && result.y[i] == data.y[j]) {
        found = true;
        break;
      }
    }

    ASSERT_TRUE(found) << "Expected to find row [" << result.x.row(i) << "] in the original data: " << std::endl << data.x << std::endl;
  }
}

TEST(StratifiedProportionalSample, TwoGroupsDifferentSizeEvenSampleSize) {
  Random::seed(0);

  Data<long double> x(9, 3);

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

  DataSpec<long double, int> data(x, y, { 0, 1 });

  DataSpec<long double, int> result = stratified_proportional_sample(data, 6).get_sample();

  std::map<int, int> result_sizes;

  for (int i = 0; i < result.y.size(); i++) {
    result_sizes[result.y[i]]++;
  }

  ASSERT_EQ(6, result.x.rows());
  ASSERT_EQ(2, result_sizes[0]);
  ASSERT_EQ(4, result_sizes[1]);

  for (int i = 0; i < result.x.rows(); i++) {
    bool found = false;

    for (int j = 0; j < data.x.rows(); j++) {
      if (result.x.row(i) == data.x.row(j) && result.y[i] == data.y[j]) {
        found = true;
        break;
      }
    }

    ASSERT_TRUE(found) << "Expected to find row [" << result.x.row(i) << "] in the original data: " << std::endl << data.x << std::endl;
  }
}

TEST(StratifiedProportionalSample, TwoGroupsDifferentSizeOddSampleSize) {
  Random::seed(0);

  Data<long double> x(9, 3);

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

  DataSpec<long double, int> data(x, y, { 0, 1 });

  DataSpec<long double, int> result = stratified_proportional_sample(data, 5).get_sample();

  std::map<int, int> result_sizes;

  for (int i = 0; i < result.y.size(); i++) {
    result_sizes[result.y[i]]++;
  }

  ASSERT_EQ(5, result.x.rows());
  ASSERT_EQ(1, result_sizes[0]);
  ASSERT_EQ(4, result_sizes[1]);

  for (int i = 0; i < result.x.rows(); i++) {
    bool found = false;

    for (int j = 0; j < data.x.rows(); j++) {
      if (result.x.row(i) == data.x.row(j) && result.y[i] == data.y[j]) {
        found = true;
        break;
      }
    }

    ASSERT_TRUE(found) << "Expected to find row [" << result.x.row(i) << "] in the original data: " << std::endl << data.x << std::endl;
  }
}

TEST(StratifiedPorportionalSample, AtLeastOneObservationPerGroup) {
  Random::seed(0);

  Data<long double> x(9, 3);

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

  DataSpec<long double, int> data(x, y, { 0, 1, 2 });

  DataSpec<long double, int> result = stratified_proportional_sample(data, 3).get_sample();

  std::map<int, int> result_sizes;

  for (int i = 0; i < result.y.size(); i++) {
    result_sizes[result.y[i]]++;
  }

  ASSERT_EQ(4, result.x.rows());
  ASSERT_EQ(1, result_sizes[0]);
  ASSERT_EQ(2, result_sizes[1]);
  ASSERT_EQ(1, result_sizes[2]);
}

TEST(BootstrapDataSpec, CenterSingleObservation) {
  Data<long double> x(1, 3);
  x <<
    1.0, 2.0, 6.0;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  BootstrapDataSpec<long double, int> actual = center(data);

  Data<long double> expected_x = Data<long double>::Zero(1, 3);

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.sample_indices, actual.sample_indices);
  ASSERT_EQ(data.classes, actual.classes);
}

TEST(BootstrapDataSpec, CenterMultipleEqualObservations) {
  Data<long double> x(3, 3);
  x <<
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  BootstrapDataSpec<long double, int> actual = center(data);

  Data<long double> expected_x = Data<long double>::Zero(3, 3);

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.sample_indices, actual.sample_indices);
  ASSERT_EQ(data.classes, actual.classes);
}

TEST(BootstrapDataSpec, CenterMultipleDifferentObservations) {
  Data<long double> x(3, 3);
  x <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  BootstrapDataSpec<long double, int> actual = center(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    -1.0, -1.0, -1.0,
    0.0, 0.0, 0.0,
    1.0, 1.0, 1.0;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.sample_indices, actual.sample_indices);
  ASSERT_EQ(data.classes, actual.classes);
}

TEST(BootstrapDataSpec, DescaleZeroMatrix) {
  Data<long double> x(3, 3);
  x <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  BootstrapDataSpec<long double, int> actual = descale(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.sample_indices, actual.sample_indices);
  ASSERT_EQ(data.classes, actual.classes);
}

TEST(BootstrapDataSpec, DescaleConstantMatrix) {
  Data<long double> x(3, 3);
  x <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  BootstrapDataSpec<long double, int> actual = descale(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.sample_indices, actual.sample_indices);
  ASSERT_EQ(data.classes, actual.classes);
}

TEST(BootstrapDataSpec, DescaleDescaledData) {
  Data<long double> x(3, 3);
  x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  BootstrapDataSpec<long double, int> actual = descale(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.sample_indices, actual.sample_indices);
  ASSERT_EQ(data.classes, actual.classes);
}

TEST(BootstrapDataSpec, DescaleScaledData) {
  Data<long double> x(3, 3);
  x <<
    2, 4, 6,
    4, 6, 8,
    6, 8, 10;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  BootstrapDataSpec<long double, int> actual = descale(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.sample_indices, actual.sample_indices);
  ASSERT_EQ(data.classes, actual.classes);
}

TEST(BootstrapDataSpec, DescalePartiallyScaledData) {
  Data<long double> x(3, 3);
  x <<
    2, 4, 3,
    4, 6, 4,
    6, 8, 5;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  BootstrapDataSpec<long double, int> actual = descale(data);

  Data<long double> expected_x(3, 3);
  expected_x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.sample_indices, actual.sample_indices);
  ASSERT_EQ(data.classes, actual.classes);
}

TEST(BootstrapDataSpec, GetSampleGeneric) {
  Data<long double> x(3, 3);
  x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  DataSpec<long double, int> sample = data.get_sample();

  Data<long double> expected_x(2, 3);
  expected_x <<
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> expected_y(2);
  expected_y <<
    2,
    3;

  std::set<int> expected_classes = { 2, 3 };

  ASSERT_EQ(expected_x.size(), sample.x.size());
  ASSERT_EQ(expected_x.rows(), sample.x.rows());
  ASSERT_EQ(expected_x.cols(), sample.x.cols());
  ASSERT_EQ(expected_x, sample.x);

  ASSERT_EQ(expected_y.size(), sample.y.size());
  ASSERT_EQ(expected_y.rows(), sample.y.rows());
  ASSERT_EQ(expected_y.cols(), sample.y.cols());
  ASSERT_EQ(expected_y, sample.y);

  ASSERT_EQ(expected_classes.size(), sample.classes.size());
  ASSERT_EQ(expected_classes, sample.classes);
}

TEST(BootstrapDataSpec, UnwrapUniqueIndices) {
  Data<long double> x(3, 3);
  x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2 });

  auto [unwrapped_x, unwrapped_y, unwrapped_classes] = data.unwrap();

  Data<long double> expected_x(2, 3);
  expected_x <<
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> expected_y(2);
  expected_y <<
    2,
    3;

  std::set<int> expected_classes = { 2, 3 };

  ASSERT_EQ(expected_x.size(), unwrapped_x.size());
  ASSERT_EQ(expected_x.rows(), unwrapped_x.rows());
  ASSERT_EQ(expected_x.cols(), unwrapped_x.cols());
  ASSERT_EQ(expected_x, unwrapped_x);

  ASSERT_EQ(expected_y.size(), unwrapped_y.size());
  ASSERT_EQ(expected_y.rows(), unwrapped_y.rows());
  ASSERT_EQ(expected_y.cols(), unwrapped_y.cols());
  ASSERT_EQ(expected_y, unwrapped_y);

  ASSERT_EQ(expected_classes.size(), unwrapped_classes.size());
  ASSERT_EQ(expected_classes, unwrapped_classes);
}

TEST(BootstrapDataSpec, UnwrapRepeatedIndices) {
  Data<long double> x(3, 3);
  x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2, 2 });

  auto [unwrapped_x, unwrapped_y, unwrapped_classes] = data.unwrap();

  Data<long double> expected_x(3, 3);
  expected_x <<
    2, 3, 4,
    3, 4, 5,
    3, 4, 5;

  DataColumn<int> expected_y(3);
  expected_y <<
    2,
    3,
    3;

  std::set<int> expected_classes = { 2, 3 };

  ASSERT_EQ(expected_x.size(), unwrapped_x.size());
  ASSERT_EQ(expected_x.rows(), unwrapped_x.rows());
  ASSERT_EQ(expected_x.cols(), unwrapped_x.cols());
  ASSERT_EQ(expected_x, unwrapped_x);

  ASSERT_EQ(expected_y.size(), unwrapped_y.size());
  ASSERT_EQ(expected_y.rows(), unwrapped_y.rows());
  ASSERT_EQ(expected_y.cols(), unwrapped_y.cols());
  ASSERT_EQ(expected_y, unwrapped_y);

  ASSERT_EQ(expected_classes.size(), unwrapped_classes.size());
  ASSERT_EQ(expected_classes, unwrapped_classes);
}

TEST(BootstrapDataSpec, OOBIndicesAssertIndicesAreComplementary) {
  Data<long double> x(4, 3);
  x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5,
    4, 5, 6;

  DataColumn<int> y(4);
  y <<
    1,
    2,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2, 2 });

  std::set<int> expected_oob_indices = { 0, 3 };

  ASSERT_EQ(data.oob_indices.size(), expected_oob_indices.size());
  ASSERT_EQ(data.oob_indices, expected_oob_indices);
}

TEST(BootstrapDataSpec, GetOOBAssertDataIsComplementary) {
  Data<long double> x(4, 3);
  x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5,
    4, 5, 6;

  DataColumn<int> y(4);
  y <<
    1,
    2,
    2,
    3;

  BootstrapDataSpec<long double, int> data(x, y, { 1, 2, 2 });

  DataSpec<long double, int> oob = data.get_oob();

  Data<long double> expected_x(2, 3);
  expected_x <<
    1, 2, 3,
    4, 5, 6;

  DataColumn<int> expected_y(2);
  expected_y <<
    1,
    3;

  std::set<int> expected_classes = { 1, 3 };

  ASSERT_EQ(expected_x.size(), oob.x.size());
  ASSERT_EQ(expected_x.rows(), oob.x.rows());
  ASSERT_EQ(expected_x.cols(), oob.x.cols());
  ASSERT_EQ(expected_x, oob.x);

  ASSERT_EQ(expected_y.size(), oob.y.size());
  ASSERT_EQ(expected_y.rows(), oob.y.rows());
  ASSERT_EQ(expected_y.cols(), oob.y.cols());
  ASSERT_EQ(expected_y, oob.y);

  ASSERT_EQ(expected_classes.size(), oob.classes.size());
  ASSERT_EQ(expected_classes, oob.classes);
}
