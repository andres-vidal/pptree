#include <gtest/gtest.h>

#include "DataSpec.hpp"

using namespace models::stats;

TEST(DataSpec, CenterSingleObservation) {
  Data<float> x(1, 3);
  x <<
    1.0, 2.0, 6.0;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<float, int> data(x, y);

  DataSpec<float, int> actual = center(data);

  Data<float> expected_x = Data<float>::Zero(1, 3);

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(DataSpec, CenterMultipleEqualObservations) {
  Data<float> x(3, 3);
  x <<
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0,
    1.0, 2.0, 6.0;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<float, int> data(x, y);

  DataSpec<float, int> actual = center(data);

  Data<float> expected_x = Data<float>::Zero(3, 3);

  ASSERT_EQ(expected_x.size(), actual.x.size());
  ASSERT_EQ(expected_x.rows(), actual.x.rows());
  ASSERT_EQ(expected_x.cols(), actual.x.cols());
  ASSERT_EQ(expected_x, actual.x);

  ASSERT_EQ(data.y.size(), actual.y.size());
  ASSERT_EQ(data.y.rows(), actual.y.rows());
  ASSERT_EQ(data.y.cols(), actual.y.cols());
  ASSERT_EQ(data.y, actual.y);

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(DataSpec, CenterMultipleDifferentObservations) {
  Data<float> x(3, 3);
  x <<
    1.0, 2.0, 6.0,
    2.0, 3.0, 7.0,
    3.0, 4.0, 8.0;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<float, int> data(x, y);

  DataSpec<float, int> actual = center(data);

  Data<float> expected_x(3, 3);
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

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(DataSpec, DescaleZeroMatrix) {
  Data<float> x(3, 3);
  x <<
    0, 0, 0,
    0, 0, 0,
    0, 0, 0;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<float, int> data(x, y, { 1, 2 });

  DataSpec<float, int> actual = descale(data);

  Data<float> expected_x(3, 3);
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

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(DataSpec, DescaleConstantMatrix) {
  Data<float> x(3, 3);
  x <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<float, int> data(x, y, { 1, 2 });

  DataSpec<float, int> actual = descale(data);

  Data<float> expected_x(3, 3);
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

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(DataSpec, DescaleDescaledData) {
  Data<float> x(3, 3);
  x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<float, int> data(x, y, { 1, 2 });

  DataSpec<float, int> actual = descale(data);

  Data<float> expected_x(3, 3);
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

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(DataSpec, DescaleScaledData) {
  Data<float> x(3, 3);
  x <<
    2, 4, 6,
    4, 6, 8,
    6, 8, 10;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<float, int> data(x, y, { 1, 2 });

  DataSpec<float, int> actual = descale(data);

  Data<float> expected_x(3, 3);
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

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(DataSpec, DescalePartiallyScaledData) {
  Data<float> x(3, 3);
  x <<
    2, 4, 3,
    4, 6, 4,
    6, 8, 5;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<float, int> data(x, y, { 1, 2 });

  DataSpec<float, int> actual = descale(data);

  Data<float> expected_x(3, 3);
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

  ASSERT_EQ(data.classes, actual.classes);
}

TEST(DataSpec, UnwrapGeneric) {
  Data<float> x(3, 3);
  x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> y(3);
  y <<
    1,
    2,
    3;

  DataSpec<float, int> data(x, y);

  auto [unwrapped_x, unwrapped_y, unwrapped_classes] = data.unwrap();

  Data<float> expected_x(3, 3);
  expected_x <<
    1, 2, 3,
    2, 3, 4,
    3, 4, 5;

  DataColumn<int> expected_y(3);
  expected_y <<
    1,
    2,
    3;

  std::set<int> expected_classes = { 1, 2, 3 };

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
