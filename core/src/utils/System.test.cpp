/**
 * @file System.test.cpp
 * @brief Unit tests for system utilities.
 */
#include <gtest/gtest.h>

#include "utils/System.hpp"

/* The process must report a positive peak RSS value. */
TEST(PeakRSS, ReturnsPositiveValue) {
  long rss = ppforest2::sys::get_peak_rss_bytes();
  EXPECT_GT(rss, 0);
}
