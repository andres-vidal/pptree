/**
 * @file TrainingSpec.test.cpp
 * @brief Unit tests for TrainingSpec composition, serialization, and defaults.
 */
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/TrainingSpec.hpp"

using namespace ppforest2;
using json = nlohmann::json;

// ---------------------------------------------------------------------------
// is_forest()
// ---------------------------------------------------------------------------

TEST(TrainingSpec, IsForestTrue) {
  auto spec = TrainingSpec::make(pp::pda(0.3f), dr::uniform(2), sr::mean_of_means(), 5, 0, 2, 3);
  EXPECT_TRUE(spec->is_forest());
}

TEST(TrainingSpec, IsForestFalse) {
  auto spec = TrainingSpec::make(pp::pda(0.3f), dr::uniform(2), sr::mean_of_means(), 0, 0, 2, 3);
  EXPECT_FALSE(spec->is_forest());
}

// ---------------------------------------------------------------------------
// resolve_threads()
// ---------------------------------------------------------------------------

TEST(TrainingSpec, ResolveThreadsExplicit) {
  auto spec = TrainingSpec::make(pp::pda(0.3f), dr::uniform(2), sr::mean_of_means(), 5, 0, 4, 3);
  EXPECT_EQ(spec->resolve_threads(), 4);
}

TEST(TrainingSpec, ResolveThreadsDefault) {
  auto spec = TrainingSpec::make(pp::pda(0.3f), dr::uniform(2), sr::mean_of_means(), 5, 0, 0, 3);
  EXPECT_GT(spec->resolve_threads(), 0);
}

// ---------------------------------------------------------------------------
// to_json / from_json round-trip
// ---------------------------------------------------------------------------

TEST(TrainingSpec, ToJsonRoundTrip) {
  auto spec = TrainingSpec::make(pp::pda(0.3f), dr::uniform(2), sr::mean_of_means(), 5, 0, 2, 5);

  json j;
  spec->to_json(j);
  auto restored = TrainingSpec::from_json(j);

  EXPECT_EQ(restored->size, 5);
  EXPECT_EQ(restored->seed, 0);
  EXPECT_EQ(restored->threads, 2);
  EXPECT_EQ(restored->max_retries, 5);

  json j2;
  restored->to_json(j2);
  EXPECT_EQ(j, j2);
}

TEST(TrainingSpec, ToJsonRoundTripSingleTree) {
  auto spec = TrainingSpec::make(pp::pda(0.3f), dr::noop(), sr::mean_of_means(), 0, 0, 0, 3);

  json j;
  spec->to_json(j);
  auto restored = TrainingSpec::from_json(j);

  EXPECT_EQ(restored->size, 0);
  EXPECT_EQ(restored->seed, 0);
  EXPECT_EQ(restored->threads, 0);
  EXPECT_EQ(restored->max_retries, 3);

  json j2;
  restored->to_json(j2);
  EXPECT_EQ(j, j2);
}

// ---------------------------------------------------------------------------
// from_json — default values for optional fields
// ---------------------------------------------------------------------------

TEST(TrainingSpec, FromJsonDefaultsOptionalFields) {
  json j = {{"pp", {{"name", "pda"}, {"lambda", 0.3}}},
            {"dr", {{"name", "uniform"}, {"n_vars", 2}}},
            {"sr", {{"name", "mean_of_means"}}}};

  auto spec = TrainingSpec::from_json(j);

  EXPECT_EQ(spec->size, 0);
  EXPECT_EQ(spec->seed, 0);
  EXPECT_EQ(spec->threads, 0);
  EXPECT_EQ(spec->max_retries, 3);
}

// ---------------------------------------------------------------------------
// from_json — missing required strategy blocks
// ---------------------------------------------------------------------------

TEST(TrainingSpec, FromJsonMissingPPThrows) {
  json j = {{"dr", {{"name", "uniform"}, {"n_vars", 2}}}, {"sr", {{"name", "mean_of_means"}}}};

  EXPECT_THROW(TrainingSpec::from_json(j), std::exception);
}

TEST(TrainingSpec, FromJsonMissingDRThrows) {
  json j = {{"pp", {{"name", "pda"}, {"lambda", 0.3}}}, {"sr", {{"name", "mean_of_means"}}}};

  EXPECT_THROW(TrainingSpec::from_json(j), std::exception);
}

TEST(TrainingSpec, FromJsonMissingSRThrows) {
  json j = {{"pp", {{"name", "pda"}, {"lambda", 0.3}}}, {"dr", {{"name", "uniform"}, {"n_vars", 2}}}};

  EXPECT_THROW(TrainingSpec::from_json(j), std::exception);
}

// ---------------------------------------------------------------------------
// display_name must not leak into JSON serialization
// ---------------------------------------------------------------------------

TEST(TrainingSpec, DisplayNameNotInJson) {
  auto spec = TrainingSpec::make(pp::pda(0.3f), dr::uniform(2), sr::mean_of_means(), 5, 0, 2, 3);

  json j;
  spec->to_json(j);

  EXPECT_FALSE(j["pp"].contains("display_name"));
  EXPECT_FALSE(j["dr"].contains("display_name"));
  EXPECT_FALSE(j["sr"].contains("display_name"));
}

// ---------------------------------------------------------------------------
// to_json — strategy fields are complete
// ---------------------------------------------------------------------------

TEST(TrainingSpec, ToJsonContainsAllStrategyFields) {
  auto spec = TrainingSpec::make(pp::pda(0.3f), dr::uniform(2), sr::mean_of_means(), 5, 0, 2, 3);

  json j;
  spec->to_json(j);

  EXPECT_TRUE(j["pp"].contains("name"));
  EXPECT_TRUE(j["pp"].contains("lambda"));
  EXPECT_EQ(j["pp"]["name"], "pda");

  EXPECT_TRUE(j["dr"].contains("name"));
  EXPECT_TRUE(j["dr"].contains("n_vars"));
  EXPECT_EQ(j["dr"]["name"], "uniform");

  EXPECT_TRUE(j["sr"].contains("name"));
  EXPECT_EQ(j["sr"]["name"], "mean_of_means");
}
