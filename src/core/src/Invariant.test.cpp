#include "Invariant.hpp"

#include <gtest/gtest.h>

TEST(Invariant, DoesNotThrowWhenConditionIsTrue) {
  ASSERT_NO_THROW(invariant(true, "This should not throw"));
}

TEST(Invariant, ThrowsWhenConditionIsFalse) {
  ASSERT_THROW({
    try {
      invariant(false, "This should throw");
    } catch (const std::runtime_error &e) {
      EXPECT_STREQ("This should throw", e.what());
      throw;
    }
  }, std::runtime_error);
}
