#include <gtest/gtest.h>

#include "DVector.hpp"

#include "Macros.hpp"

using namespace models::math;


TEST(DVector, CollinearCollinearSameDirection) {
  DVector<float> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<float> b(3);
  b << 2.0, 4.0, 12.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(DVector, CollinearCollinearOppositeDirection) {
  DVector<float> a(3);
  a << 1.0, 2.0, 6.0;

  DVector<float> b(3);
  b << -1.0, -2.0, -6.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(DVector, CollinearNonCollinear) {
  DVector<float> a(3);
  a << 0.0, 1.0, 0.0;

  DVector<float> b(3);
  b << 1.0, 0.0, 0.0;

  ASSERT_FALSE(collinear(a, b));
}
