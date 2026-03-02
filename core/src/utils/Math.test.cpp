#include <gtest/gtest.h>

#include "utils/Math.hpp"
#include "utils/Macros.hpp"

using namespace pptree;
using namespace pptree::types;
using namespace pptree::math;

TEST(Collinear, CollinearSameDirection) {
  Vector<float> a(3);
  a << 1.0, 2.0, 6.0;

  Vector<float> b(3);
  b << 2.0, 4.0, 12.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(Collinear, CollinearOppositeDirection) {
  Vector<float> a(3);
  a << 1.0, 2.0, 6.0;

  Vector<float> b(3);
  b << -1.0, -2.0, -6.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(Collinear, NonCollinear) {
  Vector<float> a(3);
  a << 0.0, 1.0, 0.0;

  Vector<float> b(3);
  b << 1.0, 0.0, 0.0;

  ASSERT_FALSE(collinear(a, b));
}
