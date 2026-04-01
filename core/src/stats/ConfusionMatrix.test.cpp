/**
 * @file ConfusionMatrix.test.cpp
 * @brief Unit tests for the ConfusionMatrix class.
 *
 * Covers matrix construction, overall and per-group error computation,
 * JSON serialization (structure and absence of removed fields), and
 * formatted terminal output (header, diagonal, marginal errors).
 */
#include <gtest/gtest.h>

#include "stats/ConfusionMatrix.hpp"
#include "utils/Types.hpp"
#include "io/Color.hpp"
#include "io/Presentation.hpp"
#include "serialization/Json.hpp"

#include "utils/Macros.hpp"

#include <sstream>
#include <nlohmann/json.hpp>

using namespace ppforest2;
using namespace ppforest2::stats;
using namespace ppforest2::types;


TEST(ConfusionMatrix, Identity) {
  ResponseVector actual   = VEC(Response, 0, 1, 2);
  ResponseVector expected = VEC(Response, 0, 1, 2);

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Matrix<int> expected_result_values = MAT(Response, rows(3), 1, 0, 0, 0, 1, 0, 0, 0, 1);

  ASSERT_EQ(expected_result_values.size(), result.values.size());
  ASSERT_EQ(expected_result_values.rows(), result.values.rows());
  ASSERT_EQ(expected_result_values.cols(), result.values.cols());
  ASSERT_EQ(expected_result_values, result.values);
}

/* Uneven group sizes with perfect predictions give a weighted diagonal. */
TEST(ConfusionMatrix, Diagonal) {
  ResponseVector actual   = VEC(Response, 0, 1, 1, 2, 2, 2);
  ResponseVector expected = VEC(Response, 0, 1, 1, 2, 2, 2);

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Matrix<int> expected_result_values = MAT(Response, rows(3), 1, 0, 0, 0, 2, 0, 0, 0, 3);

  ASSERT_EQ(expected_result_values.size(), result.values.size());
  ASSERT_EQ(expected_result_values.rows(), result.values.rows());
  ASSERT_EQ(expected_result_values.cols(), result.values.cols());
  ASSERT_EQ(expected_result_values, result.values);

  ASSERT_EQ((std::map<int, int>({{0, 0}, {1, 1}, {2, 2}})), result.label_index);
}


TEST(ConfusionMatrix, InverseDiagonal) {
  ResponseVector actual   = VEC(Response, 0, 1, 2);
  ResponseVector expected = VEC(Response, 2, 1, 0);

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Matrix<int> expected_result_values = MAT(Response, rows(3), 0, 0, 1, 0, 1, 0, 1, 0, 0);

  ASSERT_EQ(expected_result_values.size(), result.values.size());
  ASSERT_EQ(expected_result_values.rows(), result.values.rows());
  ASSERT_EQ(expected_result_values.cols(), result.values.cols());
  ASSERT_EQ(expected_result_values, result.values);

  ASSERT_EQ((std::map<int, int>({{0, 0}, {1, 1}, {2, 2}})), result.label_index);
}

/* Every prediction is shifted by one group — zero diagonal. */
TEST(ConfusionMatrix, ZeroDiagonal) {
  ResponseVector actual   = VEC(Response, 0, 1, 2);
  ResponseVector expected = VEC(Response, 1, 2, 0);

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Matrix<int> expected_result_values = MAT(Response, rows(3), 0, 0, 1, 1, 0, 0, 0, 1, 0);

  ASSERT_EQ(expected_result_values.size(), result.values.size());
  ASSERT_EQ(expected_result_values.rows(), result.values.rows());
  ASSERT_EQ(expected_result_values.cols(), result.values.cols());
  ASSERT_EQ(expected_result_values, result.values);

  ASSERT_EQ((std::map<int, int>({{0, 0}, {1, 1}, {2, 2}})), result.label_index);
}

/* Mixed predictions with one misclassification in group 2. */
TEST(ConfusionMatrix, Generic) {
  ResponseVector actual   = VEC(Response, 0, 1, 1, 2, 2, 2);
  ResponseVector expected = VEC(Response, 0, 1, 1, 2, 2, 0);

  ConfusionMatrix result = ConfusionMatrix(actual, expected);

  Matrix<int> expected_result_values = MAT(Response, rows(3), 1, 0, 1, 0, 2, 0, 0, 0, 2);

  ASSERT_EQ(expected_result_values.size(), result.values.size());
  ASSERT_EQ(expected_result_values.rows(), result.values.rows());
  ASSERT_EQ(expected_result_values.cols(), result.values.cols());
  ASSERT_EQ(expected_result_values, result.values);

  ASSERT_EQ((std::map<int, int>({{0, 0}, {1, 1}, {2, 2}})), result.label_index);
}


TEST(ConfusionMatrix, MorePredictionsThanObservations) {
  ResponseVector predictions  = VEC(Response, 0, 1, 2);
  ResponseVector observations = VEC(Response, 0, 1);

  ASSERT_THROW(ConfusionMatrix(predictions, observations), std::invalid_argument);
}

TEST(ConfusionMatrix, MoreObservationsThanPredictions) {
  ResponseVector predictions  = VEC(Response, 0, 1);
  ResponseVector observations = VEC(Response, 0, 1, 2);

  ASSERT_THROW(ConfusionMatrix(predictions, observations), std::invalid_argument);
}

TEST(ConfusionMatrix, NonConsecutiveLabels) {
  ResponseVector actual      = VEC(Response, 1, 1, 3, 3, 5, 5);
  ResponseVector predictions = VEC(Response, 1, 3, 3, 5, 5, 1);

  ConfusionMatrix result = ConfusionMatrix(predictions, actual);

  ASSERT_EQ(3, result.values.rows());
  ASSERT_EQ(3, result.values.cols());

  ASSERT_EQ((std::map<int, int>({{1, 0}, {3, 1}, {5, 2}})), result.label_index);


  ASSERT_EQ(1, result.values(0, 0));
  ASSERT_EQ(1, result.values(0, 1));
  ASSERT_EQ(0, result.values(0, 2));

  ASSERT_NEAR(3.0 / 6.0, result.error(), 0.001);
}

// ---------------------------------------------------------------------------
// Overall error rate — error()
// ---------------------------------------------------------------------------

/* Perfect predictions -> 0% error. */
TEST(ConfusionMatrix, ErrorMin) {
  ResponseVector actual      = VEC(Response, 0, 1, 2);
  ResponseVector predictions = VEC(Response, 0, 1, 2);

  float result = ConfusionMatrix(predictions, actual).error();

  ASSERT_FLOAT_EQ(0, result);
}

/* All predictions wrong -> 100% error. */
TEST(ConfusionMatrix, ErrorMax) {
  ResponseVector actual      = VEC(Response, 0, 1, 2);
  ResponseVector predictions = VEC(Response, 2, 0, 1);

  float result = ConfusionMatrix(predictions, actual).error();

  ASSERT_FLOAT_EQ(1, result);
}

/* Half of observations misclassified -> 50% error. */
TEST(ConfusionMatrix, ErrorGeneric) {
  ResponseVector actual      = VEC(Response, 0, 1, 1, 2, 2, 2);
  ResponseVector predictions = VEC(Response, 0, 1, 2, 0, 1, 2);

  float result = ConfusionMatrix(predictions, actual).error();

  ASSERT_NEAR(0.5, result, 0.0001);
}

// ---------------------------------------------------------------------------
// Per-group error rates — group_errors()
// ---------------------------------------------------------------------------

/* Perfect predictions -> every group has 0% error. */
TEST(ConfusionMatrix, ClassErrorsAllZero) {
  ResponseVector actual      = VEC(Response, 0, 1, 2);
  ResponseVector predictions = VEC(Response, 0, 1, 2);

  FeatureVector result = ConfusionMatrix(predictions, actual).group_errors();

  FeatureVector expected_errors = VEC(Feature, 0, 0, 0);

  ASSERT_EQ(expected_errors.size(), result.size());
  ASSERT_EQ(expected_errors, result);
}

/* Every single prediction is wrong -> every group has 100% error. */
TEST(ConfusionMatrix, ClassErrorsAllOne) {
  ResponseVector actual      = VEC(Response, 0, 1, 1, 2, 2, 2);
  ResponseVector predictions = VEC(Response, 1, 2, 2, 1, 1, 1);

  FeatureVector result = ConfusionMatrix(predictions, actual).group_errors();

  FeatureVector expected_errors = VEC(Feature, 1, 1, 1);

  ASSERT_EQ(expected_errors.size(), result.size());
  ASSERT_EQ(expected_errors, result);
}

/* Mixed results: group 0 perfect, group 1 at 50%, group 2 at ~67%. */
TEST(ConfusionMatrix, ClassErrorsMixed) {
  ResponseVector actual      = VEC(Response, 0, 1, 1, 2, 2, 2);
  ResponseVector predictions = VEC(Response, 0, 1, 2, 0, 1, 2);

  FeatureVector result = ConfusionMatrix(predictions, actual).group_errors();

  FeatureVector expected_errors = VEC(Feature, 0, 0.5, 0.666667);

  ASSERT_EQ(expected_errors.size(), result.size());
  ASSERT_APPROX(expected_errors, result);
}

// ---------------------------------------------------------------------------
// JSON serialization — to_json()
//
// The JSON contains "matrix", "labels", and "group_errors".
// Importantly, the top-level "error" key was removed (error_rate is
// stored at the predict-result level instead).
// ---------------------------------------------------------------------------

/* Identity matrix: verify all expected keys present and "error" absent. */
TEST(ConfusionMatrix, ToJsonIdentity) {
  ResponseVector actual      = VEC(Response, 0, 1, 2);
  ResponseVector predictions = VEC(Response, 0, 1, 2);

  auto j = serialization::to_json(ConfusionMatrix(predictions, actual));

  EXPECT_TRUE(j.contains("matrix"));
  EXPECT_TRUE(j.contains("labels"));
  EXPECT_TRUE(j.contains("group_errors"));
  EXPECT_FALSE(j.contains("error"));

  auto matrix = j["matrix"];
  EXPECT_EQ(matrix.size(), 3u);
  EXPECT_EQ(matrix[0][0], 1);
  EXPECT_EQ(matrix[0][1], 0);
  EXPECT_EQ(matrix[1][1], 1);
  EXPECT_EQ(matrix[2][2], 1);
}

/* Confirm "error" key is absent for a non-trivial confusion matrix. */
TEST(ConfusionMatrix, ToJsonNoErrorKey) {
  ResponseVector actual      = VEC(Response, 0, 1, 1, 2, 2, 2);
  ResponseVector predictions = VEC(Response, 0, 1, 2, 0, 1, 2);

  auto j = serialization::to_json(ConfusionMatrix(predictions, actual));

  EXPECT_FALSE(j.contains("error"));
  EXPECT_TRUE(j.contains("matrix"));
  EXPECT_TRUE(j.contains("labels"));
  EXPECT_TRUE(j.contains("group_errors"));
}

/* Verify the actual group_errors values in the serialized JSON. */
TEST(ConfusionMatrix, ToJsonClassErrorsValues) {
  ResponseVector actual      = VEC(Response, 0, 1, 1, 2, 2, 2);
  ResponseVector predictions = VEC(Response, 0, 1, 2, 0, 1, 2);

  auto j = serialization::to_json(ConfusionMatrix(predictions, actual));

  auto ce = j["group_errors"];
  EXPECT_EQ(ce.size(), 3u);
  EXPECT_FLOAT_EQ(ce[0].get<float>(), 0.0f);
  EXPECT_NEAR(ce[1].get<float>(), 0.5f, 0.001f);
  EXPECT_NEAR(ce[2].get<float>(), 0.6667f, 0.01f);
}

/* Verify the labels array in the serialized JSON is sorted and complete. */
TEST(ConfusionMatrix, ToJsonLabels) {
  ResponseVector actual      = VEC(Response, 0, 1, 2);
  ResponseVector predictions = VEC(Response, 0, 1, 2);

  auto j = serialization::to_json(ConfusionMatrix(predictions, actual));

  auto labels = j["labels"];
  EXPECT_EQ(labels.size(), 3u);
  EXPECT_EQ(labels[0], 0);
  EXPECT_EQ(labels[1], 1);
  EXPECT_EQ(labels[2], 2);
}

/* Generic case: labels array has expected size. */
TEST(ConfusionMatrix, ToJsonGeneric) {
  ResponseVector actual      = VEC(Response, 0, 1, 1, 2, 2, 2);
  ResponseVector predictions = VEC(Response, 0, 1, 2, 0, 1, 2);

  auto j = serialization::to_json(ConfusionMatrix(predictions, actual));

  EXPECT_EQ(j["labels"].size(), 3u);
}

// ---------------------------------------------------------------------------
// Formatted terminal output — print()
//
// print() writes to stdout with:
//   - "Confusion Matrix:" header
//   - "Error" column header for per-row marginal errors
//   - Per-row error percentages (e.g. "0.0%", "50.0%")
//   - Colored diagonal cells (bold + green)
// ---------------------------------------------------------------------------

/* Basic print: outputs the header and matrix values. */
TEST(ConfusionMatrix, Print) {
  ResponseVector actual      = VEC(Response, 0, 1, 2);
  ResponseVector predictions = VEC(Response, 0, 1, 2);

  ConfusionMatrix cm(predictions, actual);

  testing::internal::CaptureStdout();
  ppforest2::io::Output out(false);
  io::print_confusion_matrix(out, cm);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("Confusion Matrix:"), std::string::npos);
  EXPECT_NE(output.find("1"), std::string::npos);
}

/* The "Error" column header appears in the printed output. */
TEST(ConfusionMatrix, PrintIncludesErrorHeader) {
  ResponseVector actual      = VEC(Response, 0, 1, 2);
  ResponseVector predictions = VEC(Response, 0, 1, 2);

  ConfusionMatrix cm(predictions, actual);

  testing::internal::CaptureStdout();
  ppforest2::io::Output out(false);
  io::print_confusion_matrix(out, cm);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("Error"), std::string::npos);
}

/* Each row shows a marginal error percentage matching group_errors(). */
TEST(ConfusionMatrix, PrintIncludesPerRowError) {
  ResponseVector actual      = VEC(Response, 0, 1, 1, 2, 2, 2);
  ResponseVector predictions = VEC(Response, 0, 1, 2, 0, 1, 2);

  ConfusionMatrix cm(predictions, actual);

  testing::internal::CaptureStdout();
  ppforest2::io::Output out(false);
  io::print_confusion_matrix(out, cm);
  std::string output = testing::internal::GetCapturedStdout();

  // Each row should have a percentage error marker
  EXPECT_NE(output.find("%"), std::string::npos);
  EXPECT_NE(output.find("0.0%"), std::string::npos);  // Class 0: 0% error
  EXPECT_NE(output.find("50.0%"), std::string::npos); // Class 1: 50% error
  EXPECT_NE(output.find("66.7%"), std::string::npos); // Class 2: 66.7% error
}

/* Perfect predictions: every row shows "0.0%" (three occurrences total). */
TEST(ConfusionMatrix, PrintPerfectPrediction) {
  ResponseVector actual      = VEC(Response, 0, 1, 2);
  ResponseVector predictions = VEC(Response, 0, 1, 2);

  ConfusionMatrix cm(predictions, actual);

  testing::internal::CaptureStdout();
  ppforest2::io::Output out(false);
  io::print_confusion_matrix(out, cm);
  std::string output = testing::internal::GetCapturedStdout();

  // All rows should show 0.0% error
  // Count occurrences of "0.0%"
  size_t count = 0;
  size_t pos   = 0;

  while ((pos = output.find("0.0%", pos)) != std::string::npos) {
    count++;
    pos += 4;
  }

  EXPECT_EQ(count, 3u);
}
