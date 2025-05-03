#include <gtest/gtest.h>

#include "BootstrapDataSpec.hpp"

using namespace models::stats;

TEST(BootstrapDataSpec, GetSampleGeneric) {
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

  BootstrapDataSpec<float, int> data(x, y, { 1, 2 });

  SortedDataSpec<float, int> sample = data.get_sample();

  Data<float> expected_x(2, 3);
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

TEST(BootstrapDataSpec, OOBIndicesAssertIndicesAreComplementary) {
  Data<float> x(4, 3);
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

  BootstrapDataSpec<float, int> data(x, y, { 1, 2, 2 });

  std::set<int> expected_oob_indices = { 0, 3 };

  ASSERT_EQ(data.oob_indices.size(), expected_oob_indices.size());
  ASSERT_EQ(data.oob_indices, expected_oob_indices);
}

TEST(BootstrapDataSpec, GetOOBAssertDataIsComplementary) {
  Data<float> x(4, 3);
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

  BootstrapDataSpec<float, int> data(x, y, { 1, 2, 2 });

  SortedDataSpec<float, int> oob = data.get_oob();

  Data<float> expected_x(2, 3);
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
