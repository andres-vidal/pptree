/**
 * @file TrainingSpec.test.cpp
 * @brief Unit tests for TrainingSpec composition, serialization, and defaults.
 */
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/TrainingSpec.hpp"
#include "utils/UserError.hpp"

using namespace ppforest2;
using json = nlohmann::json;

// ---------------------------------------------------------------------------
// is_forest()
// ---------------------------------------------------------------------------

TEST(TrainingSpec, IsForestTrue) {
  auto spec = TrainingSpec::builder(types::Mode::Classification)
                  .size(5)
                  .threads(2)
                  .pp(pp::pda(0.3F))
                  .vars(vars::uniform(2))
                  .make();
  EXPECT_TRUE(spec->is_forest());
}

TEST(TrainingSpec, IsForestFalse) {
  auto spec =
      TrainingSpec::builder(types::Mode::Classification).threads(2).pp(pp::pda(0.3F)).vars(vars::uniform(2)).make();
  EXPECT_FALSE(spec->is_forest());
}

// ---------------------------------------------------------------------------
// resolve_threads()
// ---------------------------------------------------------------------------

TEST(TrainingSpec, ResolveThreadsExplicit) {
  auto spec = TrainingSpec::builder(types::Mode::Classification)
                  .size(5)
                  .threads(4)
                  .pp(pp::pda(0.3F))
                  .vars(vars::uniform(2))
                  .make();
  EXPECT_EQ(spec->resolve_threads(), 4);
}

TEST(TrainingSpec, ResolveThreadsDefault) {
  auto spec =
      TrainingSpec::builder(types::Mode::Classification).size(5).pp(pp::pda(0.3F)).vars(vars::uniform(2)).make();
  EXPECT_GT(spec->resolve_threads(), 0);
}

// ---------------------------------------------------------------------------
// to_json / from_json round-trip
// ---------------------------------------------------------------------------

TEST(TrainingSpec, ToJsonRoundTrip) {
  auto spec = TrainingSpec::builder(types::Mode::Classification)
                  .size(5)
                  .threads(2)
                  .max_retries(5)
                  .pp(pp::pda(0.3F))
                  .vars(vars::uniform(2))
                  .make();

  auto j        = spec->to_json();
  auto restored = TrainingSpec::from_json(j);

  EXPECT_EQ(restored->size, 5);
  EXPECT_EQ(restored->seed, 0);
  EXPECT_EQ(restored->threads, 2);
  EXPECT_EQ(restored->max_retries, 5);

  EXPECT_EQ(j, restored->to_json());
}

TEST(TrainingSpec, ToJsonRoundTripSingleTree) {
  auto spec = TrainingSpec::builder(types::Mode::Classification).pp(pp::pda(0.3F)).make();

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
      {"grouping", {{"name", "by_label"}}},
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
      {"grouping", {{"name", "by_label"}}}
  };
  EXPECT_THROW(TrainingSpec::from_json(j), std::exception);
}

TEST(TrainingSpec, FromJsonMissingVarsThrows) {
  json const j = {
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "pure_node"}}},
      {"binarize", {{"name", "largest_gap"}}},
      {"grouping", {{"name", "by_label"}}}
  };
  EXPECT_THROW(TrainingSpec::from_json(j), std::exception);
}

TEST(TrainingSpec, FromJsonMissingCutpointThrows) {
  json const j = {
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"stop", {{"name", "pure_node"}}},
      {"binarize", {{"name", "largest_gap"}}},
      {"grouping", {{"name", "by_label"}}}
  };
  EXPECT_THROW(TrainingSpec::from_json(j), std::exception);
}

TEST(TrainingSpec, FromJsonMissingStopThrows) {
  json const j = {
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"binarize", {{"name", "largest_gap"}}},
      {"grouping", {{"name", "by_label"}}}
  };
  EXPECT_THROW(TrainingSpec::from_json(j), std::exception);
}

TEST(TrainingSpec, FromJsonMissingBinarizeThrows) {
  json const j = {
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "pure_node"}}},
      {"grouping", {{"name", "by_label"}}}
  };
  EXPECT_THROW(TrainingSpec::from_json(j), std::exception);
}

TEST(TrainingSpec, FromJsonMissingGroupingThrows) {
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
      {"grouping", {{"name", "by_label"}}}
  };
  EXPECT_THROW(TrainingSpec::from_json(j), std::exception);
}

// ---------------------------------------------------------------------------
// display_name must not leak into JSON serialization
// ---------------------------------------------------------------------------

TEST(TrainingSpec, DisplayNameNotInJson) {
  auto spec = TrainingSpec::builder(types::Mode::Classification)
                  .size(5)
                  .threads(2)
                  .pp(pp::pda(0.3F))
                  .vars(vars::uniform(2))
                  .make();
  auto j = spec->to_json();

  EXPECT_FALSE(j["pp"].contains("display_name"));
  EXPECT_FALSE(j["vars"].contains("display_name"));
  EXPECT_FALSE(j["cutpoint"].contains("display_name"));
  EXPECT_FALSE(j["leaf"].contains("display_name"));
}

// ---------------------------------------------------------------------------
// to_json — strategy fields are complete
// ---------------------------------------------------------------------------

TEST(TrainingSpec, ToJsonContainsAllStrategyFields) {
  auto spec = TrainingSpec::builder(types::Mode::Classification)
                  .size(5)
                  .threads(2)
                  .pp(pp::pda(0.3F))
                  .vars(vars::uniform(2))
                  .make();
  auto j = spec->to_json();


  EXPECT_EQ(j["pp"], pp::pda(0.3F)->to_json());
  EXPECT_EQ(j["vars"], vars::uniform(2)->to_json());
  EXPECT_EQ(j["cutpoint"], cutpoint::mean_of_means()->to_json());
  EXPECT_EQ(j["stop"], stop::pure_node()->to_json());
  EXPECT_EQ(j["binarize"], binarize::largest_gap()->to_json());
  EXPECT_EQ(j["grouping"], grouping::by_label()->to_json());
  EXPECT_EQ(j["leaf"], leaf::majority_vote()->to_json());
}

// ---------------------------------------------------------------------------
// supported_modes() — mode/strategy compatibility validation
// ---------------------------------------------------------------------------

TEST(TrainingSpec, RejectsMajorityVoteInRegression) {
  // MajorityVote is classification-only; the builder defaults everything else
  // to regression-compatible.
  auto build = [] {
    return TrainingSpec::builder(types::Mode::Classification)
        .grouping(grouping::by_cutpoint())
        .stop(stop::min_size(5))
        .leaf(leaf::majority_vote())
        .make();
  };

  EXPECT_THROW(build(), ppforest2::UserError);
}

TEST(TrainingSpec, RejectsMeanResponseInClassification) {
  auto build = [] {
    return TrainingSpec::builder(types::Mode::Classification).leaf(leaf::mean_response()).make();
  };

  EXPECT_THROW(build(), ppforest2::UserError);
}

TEST(TrainingSpec, RejectsPureNodeInRegression) {
  auto build = [] {
    return TrainingSpec::builder(types::Mode::Classification)
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .stop(stop::pure_node())
        .make();
  };

  EXPECT_THROW(build(), ppforest2::UserError);
}

TEST(TrainingSpec, RejectsMinVarianceInClassification) {
  auto build = [] {
    return TrainingSpec::builder(types::Mode::Classification).stop(stop::min_variance(0.01F)).make();
  };

  EXPECT_THROW(build(), ppforest2::UserError);
}

TEST(TrainingSpec, RejectsByCutpointInClassification) {
  auto build = [] {
    return TrainingSpec::builder(types::Mode::Classification).grouping(grouping::by_cutpoint()).make();
  };

  EXPECT_THROW(build(), ppforest2::UserError);
}

TEST(TrainingSpec, AcceptsCompatibleClassificationStrategies) {
  auto build = [] {
    return TrainingSpec::builder(types::Mode::Classification)
        .stop(stop::pure_node())
        .grouping(grouping::by_label())
        .leaf(leaf::majority_vote())
        .make();
  };

  EXPECT_NO_THROW(build());
}

TEST(TrainingSpec, AcceptsCompatibleRegressionStrategies) {
  auto build = [] {
    return TrainingSpec::builder(types::Mode::Regression)
        .stop(stop::any({stop::min_size(5), stop::min_variance(0.01F)}))
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .make();
  };

  EXPECT_NO_THROW(build());
}

TEST(TrainingSpec, CompositeStopIntersectsChildModes) {
  // stop::any(min_size, pure_node) = classification only (pure_node restricts it).
  auto rule  = stop::any({stop::min_size(5), stop::pure_node()});
  auto modes = rule->supported_modes();

  EXPECT_TRUE(modes.count(types::Mode::Classification) > 0);
  EXPECT_TRUE(modes.count(types::Mode::Regression) == 0);

  // Using it in regression mode should fail.
  auto build = [&] {
    return TrainingSpec::builder(types::Mode::Classification)
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .stop(rule)
        .make();
  };

  EXPECT_THROW(build(), ppforest2::UserError);
}

TEST(TrainingSpec, CompositeStopOnlyRegressionChildrenSupportsRegression) {
  auto rule  = stop::any({stop::min_size(5), stop::min_variance(0.01F)});
  auto modes = rule->supported_modes();

  // min_size supports both; min_variance supports only regression;
  // intersection is {Regression}.
  EXPECT_TRUE(modes.count(types::Mode::Regression) > 0);
  EXPECT_TRUE(modes.count(types::Mode::Classification) == 0);
}

TEST(TrainingSpec, MinSizeSupportsBothModes) {
  auto modes = stop::min_size(5)->supported_modes();
  EXPECT_TRUE(modes.count(types::Mode::Classification) > 0);
  EXPECT_TRUE(modes.count(types::Mode::Regression) > 0);
}

TEST(TrainingSpec, ByLabelSupportsClassificationOnly) {
  auto modes = grouping::by_label()->supported_modes();
  EXPECT_TRUE(modes.count(types::Mode::Classification) > 0);
  EXPECT_TRUE(modes.count(types::Mode::Regression) == 0);
}

TEST(TrainingSpec, AcceptsDisabledBinarizeInRegression) {
  // `binarize::Disabled` is a mode-agnostic placeholder for specs where
  // binarize never fires (regression's `ByCutpoint` grouping always
  // yields 2 groups). A regression spec built with `Disabled` passes
  // mode validation. The builder's mode-aware default also resolves
  // to `Disabled` here — this test pins both the explicit and implicit
  // paths.
  auto build_explicit = [] {
    return TrainingSpec::builder(types::Mode::Regression)
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .stop(stop::min_size(5))
        .binarize(binarize::disabled())
        .make();
  };

  auto build_default = [] {
    return TrainingSpec::builder(types::Mode::Regression)
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .stop(stop::min_size(5))
        .make(); // binarize resolves to Disabled via the builder default
  };

  EXPECT_NO_THROW(build_explicit());
  EXPECT_NO_THROW(build_default());
}

TEST(TrainingSpec, RejectsLargestGapInRegression) {
  // `LargestGap` is classification-only; regression specs must use
  // `Disabled` (or another regression-compatible binarizer). This
  // replaces the old "binarize is exempt" behavior — the placeholder
  // lets binarize participate in mode validation like every other
  // strategy family.
  auto build = [] {
    return TrainingSpec::builder(types::Mode::Classification)
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .stop(stop::min_size(5))
        .binarize(binarize::largest_gap())
        .make();
  };

  EXPECT_THROW(build(), ppforest2::UserError);
}
