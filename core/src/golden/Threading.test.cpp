/**
 * @file Threading.test.cpp
 * @brief Tests that multi-threaded forest training produces identical results.
 *
 * Verifies that the combination of schedule(static) and per-iteration
 * RNG(seed, i) in Forest::train() makes results independent of thread count.
 * Tests SKIP when OpenMP is not available.
 */
#include <gtest/gtest.h>

#include "models/Forest.hpp"
#include "models/TrainingSpecUGLDA.hpp"
#include "io/IO.hpp"

using namespace pptree;
using namespace pptree::stats;

#ifndef PPTREE_DATA_DIR
#error "PPTREE_DATA_DIR must be defined"
#endif

static const std::string DATA_DIR = PPTREE_DATA_DIR;

TEST(Threading, ForestSameResultsSingleVsMulti) {
  #ifndef _OPENMP
  GTEST_SKIP() << "OpenMP not available";
  #endif

  auto data = io::read_csv_sorted(DATA_DIR + "/iris.csv");

  Forest f1 = Forest::train(TrainingSpecUGLDA(2, 0.0f), data.x, data.y, 10, 42, 1);
  Forest f4 = Forest::train(TrainingSpecUGLDA(2, 0.0f), data.x, data.y, 10, 42, 4);

  ASSERT_EQ(f1, f4) << "1-thread and 4-thread forests should be identical";
}

TEST(Threading, ForestSameResultsAcrossRuns) {
  #ifndef _OPENMP
  GTEST_SKIP() << "OpenMP not available";
  #endif

  auto data = io::read_csv_sorted(DATA_DIR + "/iris.csv");

  Forest f1 = Forest::train(TrainingSpecUGLDA(2, 0.0f), data.x, data.y, 10, 42, 4);
  Forest f2 = Forest::train(TrainingSpecUGLDA(2, 0.0f), data.x, data.y, 10, 42, 4);

  ASSERT_EQ(f1, f2) << "Two runs with same seed and thread count should be identical";
}
