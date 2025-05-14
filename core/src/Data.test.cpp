#include <gtest/gtest.h>

#include "Data.hpp"

#include "Macros.hpp"

using namespace models::stats;

TEST(Data, Standardize) {
  Data<float> data = DATA(float, 3,
      1.0, 3.0, 1.0,
      2.0, 2.0, 3.0,
      3.0, 1.0, 2.0);

  Data<float> standardized = standardize(data);

  Data<float> expected = DATA(float, 3,
      -1.0, 1.0, -1.0,
      0.0, 0.0,  1.0,
      1.0, -1.0, 0.0);

  ASSERT_EQ(expected.size(), standardized.size());
  ASSERT_EQ(expected.rows(), standardized.rows());
  ASSERT_EQ(expected.cols(), standardized.cols());
  ASSERT_EQ(expected, standardized);
}

TEST(Data, Sort) {
  Data<float> x = DATA(float, 3,
      1.0, 3.0, 1.0,
      2.0, 2.0, 3.0,
      3.0, 1.0, 2.0);

  DataColumn<int> y = DATA(int, 3, 1, 2, 1);

  sort(x, y);

  Data<float> expected_x = DATA(float, 3,
      1.0, 3.0, 1.0,
      3.0, 1.0, 2.0,
      2.0, 2.0, 3.0);

  DataColumn<int> expected_y = DATA(int, 3, 1, 1, 2);

  ASSERT_EQ_DATA(expected_x, x);
  ASSERT_EQ_DATA(expected_y, y);
}
