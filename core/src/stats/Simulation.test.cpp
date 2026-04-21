#include <gtest/gtest.h>

#include "stats/Simulation.hpp"
#include "stats/GroupPartition.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;

TEST(Simulate, CorrectDimensions) {
  RNG rng(0);
  auto data = simulate(100, 4, 3, rng);

  ASSERT_EQ(data.x.rows(), 100);
  ASSERT_EQ(data.x.cols(), 4);
  ASSERT_EQ(data.y.size(), 100);
}

TEST(Simulate, CorrectNumberOfClasses) {
  RNG rng(0);
  auto data = simulate(90, 4, 3, rng);

  ASSERT_EQ(data.groups.size(), 3);
  ASSERT_TRUE(data.groups.count(0));
  ASSERT_TRUE(data.groups.count(1));
  ASSERT_TRUE(data.groups.count(2));
}

TEST(Simulate, BalancedClasses) {
  RNG rng(0);
  auto data = simulate(90, 4, 3, rng);

  GroupPartition spec(data.y);
  ASSERT_EQ(spec.group_size(0), 30);
  ASSERT_EQ(spec.group_size(1), 30);
  ASSERT_EQ(spec.group_size(2), 30);
}

TEST(Simulate, SortedByClassLabel) {
  RNG rng(0);
  auto data = simulate(90, 4, 3, rng);

  for (int i = 1; i < data.y.size(); ++i) {
    ASSERT_LE(data.y[i - 1], data.y[i]);
  }
}

TEST(Simulate, TwoClasses) {
  RNG rng(0);
  auto data = simulate(50, 2, 2, rng);

  ASSERT_EQ(data.x.rows(), 50);
  ASSERT_EQ(data.groups.size(), 2);
}

TEST(Simulate, ManyClasses) {
  RNG rng(0);
  auto data = simulate(100, 4, 10, rng);

  ASSERT_EQ(data.groups.size(), 10);
  ASSERT_EQ(data.x.rows(), 100);
}

TEST(Simulate, SingleFeature) {
  RNG rng(0);
  auto data = simulate(50, 1, 2, rng);

  ASSERT_EQ(data.x.cols(), 1);
}

TEST(Simulate, CustomParams) {
  RNG rng(0);
  SimulationParams params;
  params.mean            = 0.0f;
  params.mean_separation = 100.0f;
  params.sd              = 1.0f;

  auto data = simulate(60, 2, 3, rng, params);

  ASSERT_EQ(data.x.rows(), 60);
  ASSERT_EQ(data.groups.size(), 3);
}

TEST(Simulate, Deterministic) {
  RNG rng1(0);
  auto data1 = simulate(50, 4, 3, rng1);

  RNG rng2(0);
  auto data2 = simulate(50, 4, 3, rng2);

  ASSERT_EQ(data1.x, data2.x);
  ASSERT_EQ(data1.y, data2.y);
}

TEST(Simulate, DifferentSeedsDifferentData) {
  RNG rng1(0);
  auto data1 = simulate(50, 4, 3, rng1);

  RNG rng2(99);
  auto data2 = simulate(50, 4, 3, rng2);

  ASSERT_NE(data1.x, data2.x);
}

TEST(Split, PreservesClassProportions) {
  RNG rng(0);
  auto data = simulate(90, 4, 3, rng);

  RNG split_rng(99);
  auto s = split(data, 0.8f, split_rng);

  ASSERT_EQ(s.tr.size() + s.te.size(), 90);

  // With 30 per group and 0.8 ratio, expect ~24 train per group
  // Total train should be ~72
  ASSERT_GE(s.tr.size(), 60);
  ASSERT_LE(s.tr.size(), 78);
}

TEST(Split, IndicesAreValid) {
  RNG rng(0);
  auto data = simulate(60, 4, 3, rng);

  RNG split_rng(0);
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
  RNG rng(0);
  auto data = simulate(60, 4, 3, rng);

  RNG split_rng(0);
  auto s = split(data, 0.7f, split_rng);

  std::set<int> train_set(s.tr.begin(), s.tr.end());
  for (int idx : s.te) {
    ASSERT_EQ(train_set.count(idx), 0);
  }
}

TEST(Split, Deterministic) {
  RNG rng1(0);
  auto data = simulate(60, 4, 3, rng1);

  RNG split_rng1(0);
  auto s1 = split(data, 0.7f, split_rng1);

  RNG split_rng2(0);
  auto s2 = split(data, 0.7f, split_rng2);

  ASSERT_EQ(s1.tr, s2.tr);
  ASSERT_EQ(s1.te, s2.te);
}

TEST(Split, RegressionProducesNonEmptySides) {
  // Regression DataPackets have `groups` empty by construction. The
  // previous stratified-only implementation would return empty tr/te
  // for regression (silently breaking `evaluate --mode regression`).
  // This locks in the empty-groups fallback path.
  RNG rng(0);
  auto data = simulate_regression(60, 4, rng);

  RNG split_rng(0);
  auto s = split(data, 0.7f, split_rng);

  ASSERT_EQ(static_cast<int>(s.tr.size() + s.te.size()), 60);
  ASSERT_GT(s.tr.size(), 0u) << "regression split must produce non-empty train";
  ASSERT_GT(s.te.size(), 0u) << "regression split must produce non-empty test";
}

TEST(Split, RegressionIndicesAreSortedPerSide) {
  // `ByCutpoint` at the root of regression training requires the
  // response to be sorted ascending; `data.y` is already sorted, so
  // sorting the index vectors preserves y-order on each side.
  RNG rng(0);
  auto data = simulate_regression(40, 3, rng);

  RNG split_rng(0);
  auto s = split(data, 0.6f, split_rng);

  for (std::size_t i = 1; i < s.tr.size(); ++i) ASSERT_LT(s.tr[i - 1], s.tr[i]);
  for (std::size_t i = 1; i < s.te.size(); ++i) ASSERT_LT(s.te[i - 1], s.te[i]);
}

TEST(Split, RegressionNoOverlap) {
  RNG rng(0);
  auto data = simulate_regression(60, 4, rng);

  RNG split_rng(0);
  auto s = split(data, 0.7f, split_rng);

  std::set<int> train_set(s.tr.begin(), s.tr.end());
  for (int idx : s.te) {
    ASSERT_EQ(train_set.count(idx), 0) << "row " << idx << " in both tr and te";
  }
}

TEST(Split, RegressionDeterministic) {
  RNG rng(0);
  auto data = simulate_regression(50, 3, rng);

  RNG split_rng1(0);
  auto s1 = split(data, 0.7f, split_rng1);
  RNG split_rng2(0);
  auto s2 = split(data, 0.7f, split_rng2);

  ASSERT_EQ(s1.tr, s2.tr);
  ASSERT_EQ(s1.te, s2.te);
}

// -----------------------------------------------------------------------------
// Regression simulation
// -----------------------------------------------------------------------------

TEST(SimulateRegression, CorrectDimensions) {
  RNG rng(0);
  auto data = simulate_regression(100, 4, rng);

  ASSERT_EQ(data.x.rows(), 100);
  ASSERT_EQ(data.x.cols(), 4);
  ASSERT_EQ(data.y.size(), 100);
  // Regression data packets do not carry discrete group labels.
  ASSERT_TRUE(data.groups.empty());
}

TEST(SimulateRegression, SortedByY) {
  RNG rng(0);
  auto data = simulate_regression(50, 3, rng);

  for (int i = 1; i < 50; ++i) {
    ASSERT_LE(data.y(i - 1), data.y(i));
  }
}

TEST(SimulateRegression, Reproducible) {
  RNG rng1(0);
  RNG rng2(0);

  auto d1 = simulate_regression(50, 3, rng1);
  auto d2 = simulate_regression(50, 3, rng2);

  ASSERT_EQ(d1.x.rows(), d2.x.rows());
  for (int i = 0; i < d1.x.rows(); ++i) {
    ASSERT_FLOAT_EQ(d1.y(i), d2.y(i));
    for (int j = 0; j < d1.x.cols(); ++j) {
      ASSERT_FLOAT_EQ(d1.x(i, j), d2.x(i, j));
    }
  }
}

TEST(SimulateRegression, RespondsToInformativeFeatures) {
  // With only 2 informative features, the first 2 columns should correlate
  // strongly with y, while the rest should not.
  RegressionSimulationParams params;
  params.n_informative = 2;
  params.noise_sd      = 0.1F;

  RNG rng(0);
  auto data = simulate_regression(500, 6, rng, params);

  // Correlation of column 0 with y should be higher than column 5.
  auto corr = [&](int col) {
    double mx = 0, my = 0;
    for (int i = 0; i < 500; ++i) {
      mx += data.x(i, col);
      my += data.y(i);
    }
    mx /= 500;
    my /= 500;

    double num = 0, dx = 0, dy = 0;
    for (int i = 0; i < 500; ++i) {
      double a = data.x(i, col) - mx;
      double b = data.y(i) - my;
      num += a * b;
      dx += a * a;
      dy += b * b;
    }

    return std::abs(num) / std::sqrt(dx * dy);
  };

  double corr_informative = corr(0);
  double corr_noise       = corr(5);

  EXPECT_GT(corr_informative, corr_noise);
  EXPECT_GT(corr_informative, 0.2);
}
