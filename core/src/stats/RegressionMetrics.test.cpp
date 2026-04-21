#include "stats/RegressionMetrics.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>

using namespace ppforest2::stats;
using namespace ppforest2::types;

class RegressionMetricsTest : public ::testing::Test {
  protected:
    OutcomeVector predictions;
    OutcomeVector actual;

    void SetUp() override {
      predictions.resize(5);
      actual.resize(5);

      actual << 1.0F, 2.0F, 3.0F, 4.0F, 5.0F;
      predictions << 1.1F, 2.2F, 2.8F, 4.1F, 4.9F;
    }
};

TEST_F(RegressionMetricsTest, MseComputesCorrectly) {
  double result = mse(predictions, actual);

  // (0.1^2 + 0.2^2 + 0.2^2 + 0.1^2 + 0.1^2) / 5 = 0.11 / 5 = 0.022
  EXPECT_NEAR(result, 0.022, 1e-4);
}

TEST_F(RegressionMetricsTest, MaeComputesCorrectly) {
  double result = mae(predictions, actual);

  // (0.1 + 0.2 + 0.2 + 0.1 + 0.1) / 5 = 0.7 / 5 = 0.14
  EXPECT_NEAR(result, 0.14, 1e-4);
}

TEST_F(RegressionMetricsTest, RSquaredComputesCorrectly) {
  double result = r_squared(predictions, actual);

  // SS_res = 0.11, SS_tot = 10.0, R^2 = 1 - 0.11/10 = 0.989
  EXPECT_NEAR(result, 0.989, 1e-2);
}

TEST_F(RegressionMetricsTest, PerfectPredictions) {
  double result_mse = mse(actual, actual);
  double result_mae = mae(actual, actual);
  double result_r2  = r_squared(actual, actual);

  EXPECT_NEAR(result_mse, 0.0, 1e-10);
  EXPECT_NEAR(result_mae, 0.0, 1e-10);
  EXPECT_NEAR(result_r2, 1.0, 1e-10);
}

TEST_F(RegressionMetricsTest, StructComputesAllMetrics) {
  RegressionMetrics metrics(predictions, actual);

  EXPECT_NEAR(metrics.mse, 0.022, 1e-4);
  EXPECT_NEAR(metrics.mae, 0.14, 1e-4);
  EXPECT_GT(metrics.r_squared, 0.9);
}

TEST_F(RegressionMetricsTest, DefaultConstructorZeroMetrics) {
  RegressionMetrics metrics;

  EXPECT_EQ(metrics.mse, 0.0);
  EXPECT_EQ(metrics.mae, 0.0);
  EXPECT_EQ(metrics.r_squared, 0.0);
}

TEST_F(RegressionMetricsTest, ThrowsOnSizeMismatch) {
  OutcomeVector short_vec(3);
  short_vec << 1.0F, 2.0F, 3.0F;

  EXPECT_THROW(mse(short_vec, actual), std::invalid_argument);
  EXPECT_THROW(mae(short_vec, actual), std::invalid_argument);
  EXPECT_THROW(r_squared(short_vec, actual), std::invalid_argument);
}

TEST_F(RegressionMetricsTest, ThrowsOnEmptyVectors) {
  OutcomeVector empty(0);

  EXPECT_THROW(mse(empty, empty), std::invalid_argument);
  EXPECT_THROW(mae(empty, empty), std::invalid_argument);
  EXPECT_THROW(r_squared(empty, empty), std::invalid_argument);
}

TEST_F(RegressionMetricsTest, SingleElement) {
  OutcomeVector single_pred(1);
  OutcomeVector single_actual(1);
  single_pred << 2.0F;
  single_actual << 3.0F;

  EXPECT_NEAR(mse(single_pred, single_actual), 1.0, 1e-6);
  EXPECT_NEAR(mae(single_pred, single_actual), 1.0, 1e-6);
  // SS_tot = 0 for single element, R^2 = 0
  EXPECT_NEAR(r_squared(single_pred, single_actual), 0.0, 1e-6);
}

TEST_F(RegressionMetricsTest, ConstantActualValues) {
  OutcomeVector constant_actual(3);
  constant_actual << 5.0F, 5.0F, 5.0F;

  OutcomeVector preds(3);
  preds << 4.0F, 5.0F, 6.0F;

  // SS_tot = 0 for constant actual, R^2 = 0
  EXPECT_NEAR(r_squared(preds, constant_actual), 0.0, 1e-6);
  EXPECT_NEAR(mse(preds, constant_actual), 2.0 / 3.0, 1e-6);
}

TEST_F(RegressionMetricsTest, NegativeValues) {
  OutcomeVector neg_actual(3);
  neg_actual << -3.0F, -1.0F, 2.0F;

  OutcomeVector neg_pred(3);
  neg_pred << -2.5F, -1.5F, 1.5F;

  double result = mse(neg_pred, neg_actual);
  // (0.5^2 + 0.5^2 + 0.5^2) / 3 = 0.75/3 = 0.25
  EXPECT_NEAR(result, 0.25, 1e-6);
}
