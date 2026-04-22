#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <numeric>

#include "models/Tree.hpp"
#include "models/ClassificationTree.hpp"
#include "models/ClassificationForest.hpp"
#include "models/RegressionTree.hpp"
#include "models/RegressionForest.hpp"
#include "models/Forest.hpp"
#include "models/TrainingSpec.hpp"
#include "models/strategies/grouping/ByCutpoint.hpp"
#include "models/strategies/leaf/MeanResponse.hpp"
#include "models/strategies/stop/MinSize.hpp"
#include "models/strategies/stop/MinVariance.hpp"
#include "models/strategies/stop/CompositeStop.hpp"
#include "stats/RegressionMetrics.hpp"
#include "stats/Stats.hpp"

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace {
  /**
   * @brief Create a simple regression dataset: y = x1 + x2 + noise.
   *
   * Returns sorted x and y (sorted ascending by y).
   */
  struct RegressionData {
    FeatureMatrix x;
    OutcomeVector y;
  };

  RegressionData make_regression_data(int n, int seed) {
    RNG rng(static_cast<uint64_t>(seed));

    FeatureMatrix x(n, 2);
    OutcomeVector y(n);

    for (int i = 0; i < n; ++i) {
      Feature x1 = static_cast<Feature>(i) / static_cast<Feature>(n);
      Feature x2 = static_cast<Feature>(n - i) / static_cast<Feature>(n);
      x(i, 0)    = x1;
      x(i, 1)    = x2;
      y(i)       = x1 + x2 + static_cast<Feature>(i % 3) * 0.01F;
    }

    // Sort by y.
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);

    std::stable_sort(order.begin(), order.end(), [&y](int a, int b) { return y(a) < y(b); });

    FeatureMatrix sorted_x(n, 2);
    OutcomeVector sorted_y(n);

    for (int i = 0; i < n; ++i) {
      sorted_x.row(i) = x.row(order[static_cast<std::size_t>(i)]);
      sorted_y(i)     = y(order[static_cast<std::size_t>(i)]);
    }

    return {sorted_x, sorted_y};
  }

  TrainingSpec make_regression_tree_spec(int seed = 0) {
    return TrainingSpec::builder(types::Mode::Regression)
        .pp(pp::pda(0.0F))
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
        .seed(seed)
        .build();
  }

  TrainingSpec make_regression_forest_spec(int size = 10, int seed = 0) {
    return TrainingSpec::builder(types::Mode::Regression)
        .pp(pp::pda(0.0F))
        .vars(vars::all())
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
        .size(size)
        .seed(seed)
        .threads(1)
        .build();
  }
}
// ---------------------------------------------------------------------------
// RegressionForest tests
// ---------------------------------------------------------------------------

TEST(RegressionForest, TrainsSuccessfully) {
  auto data = make_regression_data(40, 0);
  auto spec = make_regression_forest_spec(5);

  auto forest_ptr = Forest::train(spec, data.x, data.y);
  Forest& forest  = *forest_ptr;

  EXPECT_EQ(static_cast<int>(forest.trees.size()), 5);
}

TEST(RegressionForest, PredictsMatrix) {
  auto data = make_regression_data(40, 0);
  auto spec = make_regression_forest_spec(5);

  auto forest_ptr = Forest::train(spec, data.x, data.y);
  Forest& forest  = *forest_ptr;

  OutcomeVector preds = forest.predict(data.x);

  EXPECT_EQ(preds.size(), data.x.rows());

  for (int i = 0; i < preds.size(); ++i) {
    EXPECT_TRUE(std::isfinite(preds(i)));
  }
}

TEST(RegressionForest, PredictsMean) {
  auto data = make_regression_data(40, 0);
  auto spec = make_regression_forest_spec(5);

  auto forest_ptr = Forest::train(spec, data.x, data.y);
  Forest& forest  = *forest_ptr;

  // Forest regression prediction is the mean of tree predictions.
  OutcomeVector preds = forest.predict(data.x);

  RegressionMetrics metrics(preds, data.y);

  EXPECT_LT(metrics.mse, 1.0);
}

TEST(RegressionForest, OobPredict) {
  auto data = make_regression_data(40, 0);
  auto spec = make_regression_forest_spec(10);

  auto forest_ptr = Forest::train(spec, data.x, data.y);
  Forest& forest  = *forest_ptr;

  OutcomeVector oob_preds = forest.oob_predict(data.x);

  EXPECT_EQ(oob_preds.size(), data.x.rows());
}

TEST(RegressionForest, OobError) {
  auto data = make_regression_data(40, 0);
  auto spec = make_regression_forest_spec(10);

  auto forest_ptr = Forest::train(spec, data.x, data.y);
  auto& forest    = dynamic_cast<RegressionForest&>(*forest_ptr);

  auto oob_err = forest.oob_error(data.x, data.y);

  // OOB error is the MSE against the continuous response.
  ASSERT_TRUE(oob_err.has_value());
  EXPECT_GE(*oob_err, 0.0);

  // Sanity check: recompute by hand, should match exactly.
  OutcomeVector preds = forest.oob_predict(data.x);
  double expected_mse = 0.0;
  int valid           = 0;
  for (int i = 0; i < preds.size(); ++i) {
    if (!std::isnan(static_cast<double>(preds(i)))) {
      double diff = static_cast<double>(preds(i)) - static_cast<double>(data.y(i));
      expected_mse += diff * diff;
      valid++;
    }
  }
  expected_mse /= static_cast<double>(valid);
  EXPECT_NEAR(*oob_err, expected_mse, 1e-6);
}

TEST(RegressionForest, ProportionsNotAllowed) {
  auto data = make_regression_data(40, 0);
  auto spec = make_regression_forest_spec(5);

  auto forest_ptr = Forest::train(spec, data.x, data.y);
  Forest& forest  = *forest_ptr;

  EXPECT_THROW(forest.predict(data.x, Proportions{}), std::invalid_argument);
}

TEST(RegressionForest, Reproducible) {
  auto data  = make_regression_data(30, 0);
  auto spec1 = make_regression_forest_spec(5, 0);
  auto spec2 = make_regression_forest_spec(5, 0);

  auto f1_ptr = Forest::train(spec1, data.x, data.y);
  Forest& f1  = *f1_ptr;
  auto f2_ptr = Forest::train(spec2, data.x, data.y);
  Forest& f2  = *f2_ptr;

  OutcomeVector preds1 = f1.predict(data.x);
  OutcomeVector preds2 = f2.predict(data.x);

  for (int i = 0; i < preds1.size(); ++i) {
    EXPECT_FLOAT_EQ(preds1(i), preds2(i));
  }
}

TEST(RegressionForest, MultipleThreads) {
  auto data = make_regression_data(100, 0);

  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .vars(vars::uniform(2))
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                  .size(10)
                  .seed(0)
                  .threads(2)
                  .build();

  auto forest_ptr = Forest::train(spec, data.x, data.y);
  Forest& forest  = *forest_ptr;

  EXPECT_EQ(static_cast<int>(forest.trees.size()), 10);
}

TEST(RegressionForest, OobWithManyTrees) {
  auto data = make_regression_data(50, 0);
  auto spec = make_regression_forest_spec(50, 0);

  auto forest_ptr = Forest::train(spec, data.x, data.y);
  Forest& forest  = *forest_ptr;

  OutcomeVector oob_preds = forest.oob_predict(data.x);

  // "No OOB tree" sentinel for regression is NaN (not -1, which is a valid prediction).
  int oob_count = 0;
  for (int i = 0; i < oob_preds.size(); ++i) {
    if (!std::isnan(static_cast<double>(oob_preds(i)))) {
      oob_count++;
    }
  }

  EXPECT_GT(oob_count, 45);
}

TEST(RegressionForest, OobPredictSentinelIsNaNNotMinusOne) {
  // With very few trees on small data, some observations should have no OOB
  // tree and receive the NaN sentinel. -1 used to be the sentinel but collides
  // with valid regression predictions of -1.0.
  auto data = make_regression_data(10, 0);
  auto spec = make_regression_forest_spec(2, 0);

  auto forest_ptr = Forest::train(spec, data.x, data.y);
  Forest& forest  = *forest_ptr;

  OutcomeVector oob_preds = forest.oob_predict(data.x);

  // The sentinel (if any) must be NaN, never exactly -1.
  bool saw_sentinel = false;
  for (int i = 0; i < oob_preds.size(); ++i) {
    double v = static_cast<double>(oob_preds(i));
    if (std::isnan(v)) {
      saw_sentinel = true;
    } else {
      EXPECT_TRUE(std::isfinite(v));
    }
  }

  // With only 2 trees on 10 observations, at least one should have no OOB tree.
  // (Not strictly guaranteed, but overwhelmingly likely with seed=0.)
  EXPECT_TRUE(saw_sentinel);
}

TEST(RegressionForest, OobErrorRejectsWrongTruthSize) {
  auto data = make_regression_data(40, 0);
  auto spec = make_regression_forest_spec(5);

  auto forest_ptr = Forest::train(spec, data.x, data.y);
  auto& forest    = dynamic_cast<RegressionForest&>(*forest_ptr);

  OutcomeVector wrong_size(10);
  wrong_size.setZero();

  EXPECT_THROW(forest.oob_error(data.x, wrong_size), std::exception);
}

TEST(RegressionForest, OobErrorIncludesPredictionsThatEqualMinusOne) {
  // Construct a dataset where the forest would predict exactly -1 for some
  // observations. The old sentinel (-1) would incorrectly drop those rows
  // from oob_error; NaN sentinel at the per-row level + `std::optional`
  // at the scalar-error level both avoid that class of collision.
  int const n = 40;
  FeatureMatrix x(n, 2);
  OutcomeVector y(n);

  // y values centered around -1 so the tree's leaf means are near -1.
  for (int i = 0; i < n; ++i) {
    x(i, 0) = static_cast<Feature>(i);
    x(i, 1) = static_cast<Feature>(n - i);
    y(i)    = -1.0F + 0.01F * (static_cast<Feature>(i) - n / 2.0F);
  }

  auto spec = make_regression_forest_spec(5);

  auto forest_ptr = Forest::train(spec, x, y);
  auto& forest    = dynamic_cast<RegressionForest&>(*forest_ptr);
  auto err        = forest.oob_error(x, y);

  // Error must be a valid finite MSE, not `nullopt` (which would indicate
  // every observation got dropped).
  ASSERT_TRUE(err.has_value());
  EXPECT_GE(*err, 0.0);
}

TEST(RegressionForest, PredictsMeanOfTreePredictions) {
  auto data = make_regression_data(40, 0);
  auto spec = make_regression_forest_spec(3, 0);

  auto forest_ptr = Forest::train(spec, data.x, data.y);
  Forest& forest  = *forest_ptr;

  OutcomeVector forest_preds = forest.predict(data.x);

  for (int i = 0; i < data.x.rows(); ++i) {
    FeatureVector row = data.x.row(i);
    Feature sum       = 0;
    for (auto const& tree : forest.trees) {
      sum += static_cast<Feature>(tree->predict(row));
    }
    Feature expected_mean = sum / static_cast<Feature>(forest.trees.size());
    EXPECT_NEAR(forest_preds(i), expected_mean, 1e-4);
  }
}

