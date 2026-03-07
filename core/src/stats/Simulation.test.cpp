#include <gtest/gtest.h>

#include "stats/Simulation.hpp"
#include "stats/GroupPartition.hpp"
#include "utils/Macros.hpp"

using namespace pptree;
using namespace pptree::types;
using namespace pptree::stats;

TEST(Simulate, CorrectDimensions) {
  RNG rng(42);
  auto data = simulate(100, 4, 3, rng);

  ASSERT_EQ(data.x.rows(), 100);
  ASSERT_EQ(data.x.cols(), 4);
  ASSERT_EQ(data.y.size(), 100);
}

TEST(Simulate, CorrectNumberOfClasses) {
  RNG rng(42);
  auto data = simulate(90, 4, 3, rng);

  ASSERT_EQ(data.classes.size(), 3);
  ASSERT_TRUE(data.classes.count(0));
  ASSERT_TRUE(data.classes.count(1));
  ASSERT_TRUE(data.classes.count(2));
}

TEST(Simulate, BalancedClasses) {
  RNG rng(42);
  auto data = simulate(90, 4, 3, rng);

  GroupPartition spec(data.y);
  ASSERT_EQ(spec.group_size(0), 30);
  ASSERT_EQ(spec.group_size(1), 30);
  ASSERT_EQ(spec.group_size(2), 30);
}

TEST(Simulate, SortedByClassLabel) {
  RNG rng(42);
  auto data = simulate(90, 4, 3, rng);

  for (int i = 1; i < data.y.size(); ++i) {
    ASSERT_LE(data.y[i - 1], data.y[i]);
  }
}

TEST(Simulate, TwoClasses) {
  RNG rng(42);
  auto data = simulate(50, 2, 2, rng);

  ASSERT_EQ(data.x.rows(), 50);
  ASSERT_EQ(data.classes.size(), 2);
}

TEST(Simulate, ManyClasses) {
  RNG rng(42);
  auto data = simulate(100, 4, 10, rng);

  ASSERT_EQ(data.classes.size(), 10);
  ASSERT_EQ(data.x.rows(), 100);
}

TEST(Simulate, SingleFeature) {
  RNG rng(42);
  auto data = simulate(50, 1, 2, rng);

  ASSERT_EQ(data.x.cols(), 1);
}

TEST(Simulate, CustomParams) {
  RNG rng(42);
  SimulationParams params;
  params.mean = 0.0f;
  params.mean_separation = 100.0f;
  params.sd = 1.0f;

  auto data = simulate(60, 2, 3, rng, params);

  ASSERT_EQ(data.x.rows(), 60);
  ASSERT_EQ(data.classes.size(), 3);
}

TEST(Simulate, Deterministic) {
  RNG rng1(42);
  auto data1 = simulate(50, 4, 3, rng1);

  RNG rng2(42);
  auto data2 = simulate(50, 4, 3, rng2);

  ASSERT_EQ(data1.x, data2.x);
  ASSERT_EQ(data1.y, data2.y);
}

TEST(Simulate, DifferentSeedsDifferentData) {
  RNG rng1(42);
  auto data1 = simulate(50, 4, 3, rng1);

  RNG rng2(99);
  auto data2 = simulate(50, 4, 3, rng2);

  ASSERT_NE(data1.x, data2.x);
}

TEST(Split, PreservesClassProportions) {
  RNG rng(42);
  auto data = simulate(90, 4, 3, rng);

  RNG split_rng(42);
  auto s = split(data, 0.8f, split_rng);

  ASSERT_EQ(s.tr.size() + s.te.size(), 90);

  // With 30 per class and 0.8 ratio, expect ~24 train per class
  // Total train should be ~72
  ASSERT_GE(s.tr.size(), 60);
  ASSERT_LE(s.tr.size(), 78);
}

TEST(Split, IndicesAreValid) {
  RNG rng(42);
  auto data = simulate(60, 4, 3, rng);

  RNG split_rng(42);
  auto s = split(data, 0.7f, split_rng);

  for (int idx : s.tr) {
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, 60);
  }
  for (int idx : s.te) {
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, 60);
  }
}

TEST(Split, NoOverlapBetweenTrainAndTest) {
  RNG rng(42);
  auto data = simulate(60, 4, 3, rng);

  RNG split_rng(42);
  auto s = split(data, 0.7f, split_rng);

  std::set<int> train_set(s.tr.begin(), s.tr.end());
  for (int idx : s.te) {
    ASSERT_EQ(train_set.count(idx), 0);
  }
}

TEST(Split, Deterministic) {
  RNG rng1(42);
  auto data = simulate(60, 4, 3, rng1);

  RNG split_rng1(42);
  auto s1 = split(data, 0.7f, split_rng1);

  RNG split_rng2(42);
  auto s2 = split(data, 0.7f, split_rng2);

  ASSERT_EQ(s1.tr, s2.tr);
  ASSERT_EQ(s1.te, s2.te);
}
