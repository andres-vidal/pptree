#include <gtest/gtest.h>

#include "linear_algebra.hpp"

TEST(LinearAlgebra, sum) {
  int num1 = 1;
  int num2 = 1;
  int res = sum(num1, num2);

  ASSERT_EQ(2, res);
}
