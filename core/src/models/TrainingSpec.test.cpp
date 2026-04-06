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
  auto spec = TrainingSpec::builder().size(5).threads(2).pp(pp::pda(0.3F)).vars(vars::uniform(2)).make();
  EXPECT_TRUE(spec->is_forest());
}

TEST(TrainingSpec, IsForestFalse) {
  auto spec = TrainingSpec::builder().threads(2).pp(pp::pda(0.3F)).vars(vars::uniform(2)).make();
  EXPECT_FALSE(spec->is_forest());
}

// ---------------------------------------------------------------------------
// resolve_threads()
// ---------------------------------------------------------------------------

TEST(TrainingSpec, ResolveThreadsExplicit) {
  auto spec = TrainingSpec::builder().size(5).threads(4).pp(pp::pda(0.3F)).vars(vars::uniform(2)).make();
  EXPECT_EQ(spec->resolve_threads(), 4);
}

TEST(TrainingSpec, ResolveThreadsDefault) {
  auto spec = TrainingSpec::builder().size(5).pp(pp::pda(0.3F)).vars(vars::uniform(2)).make();
  EXPECT_GT(spec->resolve_threads(), 0);
}

// ---------------------------------------------------------------------------
// to_json / from_json round-trip
// ---------------------------------------------------------------------------

TEST(TrainingSpec, ToJsonRoundTrip) {
  auto spec = TrainingSpec::builder().size(5).threads(2).max_retries(5).pp(pp::pda(0.3F)).vars(vars::uniform(2)).make();

  auto j        = spec->to_json();
  auto restored = TrainingSpec::from_json(j);

  EXPECT_EQ(restored->size, 5);
  EXPECT_EQ(restored->seed, 0);
  EXPECT_EQ(restored->threads, 2);
  EXPECT_EQ(restored->max_retries, 5);

  EXPECT_EQ(j, restored->to_json());
}

TEST(TrainingSpec, ToJsonRoundTripSingleTree) {
  auto spec = TrainingSpec::builder().pp(pp::pda(0.3F)).make();

  auto j        = spec->to_json();
  auto restored = TrainingSpec::from_json(j);

  EXPECT_EQ(restored->size, 0);
  EXPECT_EQ(restored->seed, 0);
  EXPECT_EQ(restored->threads, 0);
  EXPECT_EQ(restored->max_retries, 3);

  EXPECT_EQ(j, restored->to_json());
}

// ---------------------------------------------------------------------------
// from_json — default values for optional fields
// ---------------------------------------------------------------------------

TEST(TrainingSpec, FromJsonDefaultsOptionalFields) {
  json const j = {
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "pure_node"}}},
      {"binarize", {{"name", "largest_gap"}}},
      {"partition", {{"name", "by_group"}}},
      {"leaf", {{"name", "majority_vote"}}}
  };

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
  json const j = {
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "pure_node"}}},
      {"binarize", {{"name", "largest_gap"}}},
      {"partition", {{"name", "by_group"}}}
  };
  EXPECT_THROW(TrainingSpec::from_json(j), std::exception);
}

TEST(TrainingSpec, FromJsonMissingVarsThrows) {
  json const j = {
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "pure_node"}}},
      {"binarize", {{"name", "largest_gap"}}},
      {"partition", {{"name", "by_group"}}}
  };
  EXPECT_THROW(TrainingSpec::from_json(j), std::exception);
}

TEST(TrainingSpec, FromJsonMissingCutpointThrows) {
  json const j = {
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"stop", {{"name", "pure_node"}}},
      {"binarize", {{"name", "largest_gap"}}},
      {"partition", {{"name", "by_group"}}}
  };
  EXPECT_THROW(TrainingSpec::from_json(j), std::exception);
}

TEST(TrainingSpec, FromJsonMissingStopThrows) {
  json const j = {
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"binarize", {{"name", "largest_gap"}}},
      {"partition", {{"name", "by_group"}}}
  };
  EXPECT_THROW(TrainingSpec::from_json(j), std::exception);
}

TEST(TrainingSpec, FromJsonMissingBinarizeThrows) {
  json const j = {
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "pure_node"}}},
      {"partition", {{"name", "by_group"}}}
  };
  EXPECT_THROW(TrainingSpec::from_json(j), std::exception);
}

TEST(TrainingSpec, FromJsonMissingPartitionThrows) {
  json const j = {
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "pure_node"}}},
      {"binarize", {{"name", "largest_gap"}}}
  };
  EXPECT_THROW(TrainingSpec::from_json(j), std::exception);
}

TEST(TrainingSpec, FromJsonMissingLeafThrows) {
  json const j = {
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "pure_node"}}},
      {"binarize", {{"name", "largest_gap"}}},
      {"partition", {{"name", "by_group"}}}
  };
  EXPECT_THROW(TrainingSpec::from_json(j), std::exception);
}

// ---------------------------------------------------------------------------
// display_name must not leak into JSON serialization
// ---------------------------------------------------------------------------

TEST(TrainingSpec, DisplayNameNotInJson) {
  auto spec = TrainingSpec::builder().size(5).threads(2).pp(pp::pda(0.3F)).vars(vars::uniform(2)).make();
  auto j    = spec->to_json();

  EXPECT_FALSE(j["pp"].contains("display_name"));
  EXPECT_FALSE(j["vars"].contains("display_name"));
  EXPECT_FALSE(j["cutpoint"].contains("display_name"));
  EXPECT_FALSE(j["leaf"].contains("display_name"));
}

// ---------------------------------------------------------------------------
// to_json — strategy fields are complete
// ---------------------------------------------------------------------------

TEST(TrainingSpec, ToJsonContainsAllStrategyFields) {
  auto spec = TrainingSpec::builder().size(5).threads(2).pp(pp::pda(0.3F)).vars(vars::uniform(2)).make();
  auto j    = spec->to_json();


  EXPECT_EQ(j["pp"], pp::pda(0.3F)->to_json());
  EXPECT_EQ(j["vars"], vars::uniform(2)->to_json());
  EXPECT_EQ(j["cutpoint"], cutpoint::mean_of_means()->to_json());
  EXPECT_EQ(j["stop"], stop::pure_node()->to_json());
  EXPECT_EQ(j["binarize"], binarize::largest_gap()->to_json());
  EXPECT_EQ(j["partition"], partition::by_group()->to_json());
  EXPECT_EQ(j["leaf"], leaf::majority_vote()->to_json());
}
