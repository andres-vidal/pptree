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
#include "models/ClassificationTree.hpp"
#include "models/ClassificationForest.hpp"
#include "models/RegressionTree.hpp"
#include "models/RegressionForest.hpp"
#include "models/TrainingSpec.hpp"
#include "io/IO.hpp"

using namespace ppforest2;
using namespace ppforest2::stats;

#ifndef PPFOREST2_DATA_DIR
#error "PPFOREST2_DATA_DIR must be defined"
#endif

static const std::string DATA_DIR = PPFOREST2_DATA_DIR;

TEST(Threading, ForestSameResultsSingleVsMulti) {
#ifndef _OPENMP
  GTEST_SKIP() << "OpenMP not available";
#endif

  auto data = io::csv::read_sorted(DATA_DIR + "/classification/iris.csv");

  auto f1_ptr = Forest::train(TrainingSpec::builder(types::Mode::Classification).size(10).threads(1).vars(vars::uniform(2)).build(), data.x, data.y);
  Forest const& f1 = *f1_ptr;
  auto f4_ptr = Forest::train(TrainingSpec::builder(types::Mode::Classification).size(10).threads(4).vars(vars::uniform(2)).build(), data.x, data.y);
  Forest const& f4 = *f4_ptr;

  ASSERT_EQ(f1, f4) << "1-thread and 4-thread forests should be identical";
}

TEST(Threading, ForestSameResultsAcrossRuns) {
#ifndef _OPENMP
  GTEST_SKIP() << "OpenMP not available";
#endif

  auto data = io::csv::read_sorted(DATA_DIR + "/classification/iris.csv");

  auto f1_ptr = Forest::train(TrainingSpec::builder(types::Mode::Classification).size(10).threads(4).vars(vars::uniform(2)).build(), data.x, data.y);
  Forest const& f1 = *f1_ptr;
  auto f2_ptr = Forest::train(TrainingSpec::builder(types::Mode::Classification).size(10).threads(4).vars(vars::uniform(2)).build(), data.x, data.y);
  Forest const& f2 = *f2_ptr;

  ASSERT_EQ(f1, f2) << "Two runs with same seed and thread count should be identical";
}
