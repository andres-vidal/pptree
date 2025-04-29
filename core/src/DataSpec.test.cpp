#include <gtest/gtest.h>

#include "DataSpec.hpp"

using namespace models::stats;

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
