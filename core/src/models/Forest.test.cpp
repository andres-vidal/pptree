#include <gtest/gtest.h>

#include "models/BootstrapTree.hpp"
#include "models/Forest.hpp"
#include "models/TreeBranch.hpp"
#include "models/TreeLeaf.hpp"

#include "models/TrainingSpec.hpp"

#include "stats/Simulation.hpp"
#include "stats/Stats.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::stats;
using namespace ppforest2::types;
using namespace ppforest2::pp;
using namespace ppforest2::math;


namespace {
  Projector as_projector(std::vector<Feature> v) {
    return Eigen::Map<Projector>(v.data(), v.size());
  }
}

TEST(Forest, TrainLDAAllVariablesProperties) {
  FeatureMatrix const x =
      MAT(Feature,
          rows(30),
          1,
          0,
          1,
          1,
          1,
          1,
          0,
          1,
          0,
          0,
          1,
          0,
          0,
          0,
          1,
          1,
          0,
          1,
          2,
          1,
          1,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1,
          0,
          1,
          0,
          0,
          2,
          1,
          1,
          0,
          1,
          1,
          2,
          1,
          0,
          0,
          2,
          0,
          1,
          0,
          2,
          1,
          0,
          2,
          5,
          0,
          0,
          1,
          2,
          5,
          0,
          0,
          2,
          3,
          5,
          1,
          0,
          2,
          2,
          5,
          1,
          0,
          1,
          2,
          5,
          0,
          1,
          1,
          2,
          5,
          0,
          1,
          2,
          2,
          5,
          2,
          1,
          1,
          2,
          5,
          1,
          1,
          1,
          2,
          5,
          1,
          1,
          2,
          2,
          5,
          2,
          1,
          2,
          2,
          5,
          1,
          2,
          1,
          2,
          5,
          2,
          1,
          1,
          9,
          8,
          0,
          0,
          1,
          9,
          8,
          0,
          0,
          2,
          9,
          8,
          1,
          0,
          2,
          9,
          8,
          1,
          0,
          1,
          9,
          8,
          0,
          1,
          1,
          9,
          8,
          0,
          1,
          2,
          9,
          8,
          2,
          1,
          1,
          9,
          8,
          1,
          1,
          1);

  OutcomeVector const y =
      VEC(Outcome, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);

  int const n_vars   = x.cols();
  float const lambda = 0;
  int const seed     = 0;

  Forest const result =
      Forest::train(TrainingSpec::builder().size(4).pp(pp::pda(lambda)).vars(vars::uniform(n_vars)).build(), x, y);

  ASSERT_EQ(result.trees.size(), 4);
  ASSERT_EQ(result.training_spec->seed, seed);

  OutcomeVector const predictions = result.predict(x);
  double const err                = error_rate(predictions, y);

  ASSERT_EQ(err, 0.0) << "Forest should achieve 0% training error on well-separated 3-group data";
}

TEST(Forest, TrainLDASomeVariablesProperties) {
  FeatureMatrix const x =
      MAT(Feature,
          rows(30),
          1,
          0,
          1,
          1,
          1,
          1,
          0,
          1,
          0,
          0,
          1,
          0,
          0,
          0,
          1,
          1,
          0,
          1,
          2,
          1,
          1,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1,
          0,
          1,
          0,
          0,
          2,
          1,
          1,
          0,
          1,
          1,
          2,
          1,
          0,
          0,
          2,
          0,
          1,
          0,
          2,
          1,
          0,
          2,
          5,
          0,
          0,
          1,
          2,
          5,
          0,
          0,
          2,
          3,
          5,
          1,
          0,
          2,
          2,
          5,
          1,
          0,
          1,
          2,
          5,
          0,
          1,
          1,
          2,
          5,
          0,
          1,
          2,
          2,
          5,
          2,
          1,
          1,
          2,
          5,
          1,
          1,
          1,
          2,
          5,
          1,
          1,
          2,
          2,
          5,
          2,
          1,
          2,
          2,
          5,
          1,
          2,
          1,
          2,
          5,
          2,
          1,
          1,
          9,
          8,
          0,
          0,
          1,
          9,
          8,
          0,
          0,
          2,
          9,
          8,
          1,
          0,
          2,
          9,
          8,
          1,
          0,
          1,
          9,
          8,
          0,
          1,
          1,
          9,
          8,
          0,
          1,
          2,
          9,
          8,
          2,
          1,
          1,
          9,
          8,
          1,
          1,
          1);

  OutcomeVector const y =
      VEC(Outcome, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);

  int const n_vars   = 2;
  float const lambda = 0;
  int const seed     = 1;


  Forest const result = Forest::train(
      TrainingSpec::builder().size(4).seed(seed).pp(pp::pda(lambda)).vars(vars::uniform(n_vars)).build(), x, y
  );

  ASSERT_EQ(result.trees.size(), 4);
  ASSERT_EQ(result.training_spec->seed, seed);

  OutcomeVector const predictions = result.predict(x);
  double const err                = error_rate(predictions, y);

  ASSERT_LT(err, 0.30) << "Forest with subset of variables should still classify well-separated data";
}

TEST(Forest, TrainPDAAllVariablesProperties) {
  FeatureMatrix const x =
      MAT(Feature,
          rows(10),
          1,
          0,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          0,
          0,
          1,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          4,
          0,
          0,
          1,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          5,
          0,
          0,
          3,
          3,
          3,
          3,
          3,
          3,
          3,
          3,
          3,
          4,
          0,
          0,
          3,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          4,
          1,
          0,
          1,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          4,
          0,
          1,
          1,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          4,
          0,
          1,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          2);

  OutcomeVector const y = VEC(Outcome, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

  int const n_vars   = x.cols();
  float const lambda = 0.1;
  int const seed     = 0;

  Forest const result =
      Forest::train(TrainingSpec::builder().size(4).pp(pp::pda(lambda)).vars(vars::uniform(n_vars)).build(), x, y);

  ASSERT_EQ(result.trees.size(), 4);
  ASSERT_EQ(result.training_spec->seed, seed);

  OutcomeVector const predictions = result.predict(x);
  double const err                = error_rate(predictions, y);

  ASSERT_EQ(err, 0.0) << "PDA forest should achieve 0% training error on well-separated 2-group data";
}

TEST(ForestSimulation, PerfectSeparationLowOOBError) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 200.0F;
  params.sd              = 1.0F;

  auto data = simulate(90, 4, 3, rng, params);

  Forest const forest =
      Forest::train(TrainingSpec::builder().size(20).threads(1).vars(vars::uniform(4)).build(), data.x, data.y);

  double const err = forest.oob_error(data.x, data.y);

  ASSERT_GE(err, 0.0);
  ASSERT_LE(err, 0.05) << "Forest should achieve near-zero OOB error on perfectly separated data";
}

TEST(ForestSimulation, HighOverlapBoundedError) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 5.0F;
  params.sd              = 50.0F;

  auto data = simulate(200, 4, 3, rng, params);

  Forest const forest =
      Forest::train(TrainingSpec::builder().size(20).threads(1).vars(vars::uniform(4)).build(), data.x, data.y);

  OutcomeVector const predictions = forest.predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.80) << "Forest error on highly overlapping data should be bounded";
}

TEST(ForestSimulation, ManyClasses) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 50.0F;
  params.sd              = 10.0F;

  auto data = simulate(300, 4, 10, rng, params);

  Forest const forest =
      Forest::train(TrainingSpec::builder().size(20).threads(1).vars(vars::uniform(4)).build(), data.x, data.y);

  OutcomeVector const predictions = forest.predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.30) << "Forest should handle 10 groups with reasonable error";
}

TEST(ForestSimulation, HighDimensionality) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 50.0F;
  params.sd              = 10.0F;

  auto data = simulate(100, 50, 3, rng, params);

  Forest const forest =
      Forest::train(TrainingSpec::builder().size(20).threads(1).vars(vars::uniform(7)).build(), data.x, data.y);

  OutcomeVector const predictions = forest.predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.20) << "Forest should handle high-dimensional data (p=50)";
}

TEST(ForestSimulation, Deterministic) {
  RNG rng(0);
  auto data = simulate(90, 4, 3, rng);

  Forest const f1 =
      Forest::train(TrainingSpec::builder().size(10).threads(1).vars(vars::uniform(4)).build(), data.x, data.y);
  Forest const f2 =
      Forest::train(TrainingSpec::builder().size(10).threads(1).vars(vars::uniform(4)).build(), data.x, data.y);

  ASSERT_EQ(f1, f2) << "Same seed should produce identical forests";
}

TEST(ForestSimulation, PDAOnOverlappingData) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 10.0F;
  params.sd              = 20.0F;

  auto data = simulate(200, 4, 3, rng, params);

  Forest const forest = Forest::train(
      TrainingSpec::builder().size(20).threads(1).pp(pp::pda(0.5F)).vars(vars::uniform(4)).build(), data.x, data.y
  );

  OutcomeVector const predictions = forest.predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.80) << "PDA forest should produce bounded error on noisy data";
}

TEST(ForestSimulation, LargeDataset) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 50.0F;
  params.sd              = 10.0F;

  auto data = simulate(2000, 10, 4, rng, params);

  Forest const forest =
      Forest::train(TrainingSpec::builder().size(10).threads(1).vars(vars::uniform(5)).build(), data.x, data.y);

  OutcomeVector const predictions = forest.predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.10) << "Forest should handle large datasets efficiently";
}

namespace {
  static Forest build_three_group_forest() {
    Forest forest;

    forest.add_tree(std::make_unique<BootstrapTree>(
        TreeBranch::make(
            as_projector({0.0, 0.0, 0.0, 0.598, -0.801}),
            -0.348,
            TreeBranch::make(
                as_projector({0.9995, 0.0, -0.031, 0.0, 0.0}), 5.553, TreeLeaf::make(1), TreeLeaf::make(2)
            ),
            TreeLeaf::make(0)
        ),
        nullptr
    ));

    forest.add_tree(std::make_unique<BootstrapTree>(
        TreeBranch::make(
            as_projector({0.9998, 0.0, -0.019, 0.0, 0.0}),
            5.300,
            TreeBranch::make(as_projector({0.999, 0.0, 0.0, 0.046, 0.0}), 1.609, TreeLeaf::make(0), TreeLeaf::make(1)),
            TreeLeaf::make(2)
        ),
        nullptr
    ));

    forest.add_tree(std::make_unique<BootstrapTree>(
        TreeBranch::make(
            as_projector({0.974, -0.226, 0.0, 0.0, 0.0}),
            3.955,
            TreeBranch::make(
                as_projector({0.0, 0.9996, -0.030, 0.0, 0.0}), 2.622, TreeLeaf::make(0), TreeLeaf::make(1)
            ),
            TreeLeaf::make(2)
        ),
        nullptr
    ));

    forest.add_tree(std::make_unique<BootstrapTree>(
        TreeBranch::make(
            as_projector({0.962, 0.0, 0.0, 0.0, -0.275}),
            4.735,
            TreeBranch::make(
                as_projector({0.0, 0.0, 0.377, 0.0, -0.926}), -0.832, TreeLeaf::make(1), TreeLeaf::make(0)
            ),
            TreeLeaf::make(2)
        ),
        nullptr
    ));

    return forest;
  }
}

TEST(Forest, PredictSingleObservation) {
  Forest const forest = build_three_group_forest();

  ASSERT_EQ(0, forest.predict(VEC(Feature, 1, 0, 1, 1, 1)));
  ASSERT_EQ(1, forest.predict(VEC(Feature, 2, 5, 0, 0, 1)));
  ASSERT_EQ(2, forest.predict(VEC(Feature, 9, 8, 1, 1, 1)));
}

TEST(Forest, PredictBatch) {
  Forest const forest = build_three_group_forest();

  FeatureMatrix const x =
      MAT(Feature, rows(6), 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 2, 5, 0, 0, 1, 2, 5, 1, 0, 1, 9, 8, 0, 0, 1, 9, 8, 1, 0, 2);

  OutcomeVector const result   = forest.predict(x);
  OutcomeVector const expected = VEC(Outcome, 0, 0, 1, 1, 2, 2);

  ASSERT_EQ(expected, result);
}

// ---------------------------------------------------------------------------
// OOB error
// ---------------------------------------------------------------------------

TEST(OobError, PerfectSeparationGivesLowError) {
  FeatureMatrix const x =
      MAT(Feature,
          rows(20),
          0.0F,
          1.0F,
          0.1F,
          2.0F,
          0.2F,
          0.5F,
          0.3F,
          1.5F,
          0.4F,
          0.8F,
          0.5F,
          1.2F,
          0.6F,
          0.9F,
          0.7F,
          1.8F,
          0.8F,
          0.6F,
          0.9F,
          1.1F,
          9.0F,
          1.0F,
          9.1F,
          2.0F,
          9.2F,
          0.5F,
          9.3F,
          1.5F,
          9.4F,
          0.8F,
          9.5F,
          1.2F,
          9.6F,
          0.9F,
          9.7F,
          1.8F,
          9.8F,
          0.6F,
          9.9F,
          1.1F);

  OutcomeVector const y = VEC(Outcome, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

  Forest const forest = Forest::train(TrainingSpec::builder().size(50).build(), x, y);

  double const err = forest.oob_error(x, y);

  ASSERT_GE(err, 0.0);
  ASSERT_LE(err, 0.1) << "Expected near-zero OOB error for perfectly separable data";
}

TEST(OobError, AllInBagReturnsNegative) {
  FeatureMatrix const x = MAT(Feature, rows(4), 0.0F, 0.0F, 0.1F, 0.1F, 9.9F, 0.0F, 9.8F, 0.1F);

  OutcomeVector const y = VEC(Outcome, 0, 0, 1, 1);

  auto condition =
      TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F);

  Forest forest;
  forest.add_tree(std::make_unique<BootstrapTree>(
      std::move(condition), TrainingSpec::builder().make(), std::vector<int>{0, 1, 2, 3}
  ));

  double const err = forest.oob_error(x, y);

  ASSERT_DOUBLE_EQ(err, -1.0) << "No OOB observations, should return -1";
}

TEST(OobError, HandBuiltTreeWithKnownOob) {
  FeatureMatrix const x = MAT(Feature, rows(6), 0.0F, 0.5F, 0.1F, 0.3F, 0.2F, 0.7F, 9.8F, 0.4F, 9.9F, 0.6F, 9.7F, 0.2F);

  OutcomeVector const y = VEC(Outcome, 0, 0, 0, 1, 1, 1);

  auto condition =
      TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F);

  Forest forest;
  forest.add_tree(std::make_unique<BootstrapTree>(
      std::move(condition), TrainingSpec::builder().make(), std::vector<int>{0, 1, 4, 5}
  ));

  double const err = forest.oob_error(x, y);

  ASSERT_DOUBLE_EQ(err, 0.0);
}

TEST(OobError, HandBuiltTreeWithOobMisclassification) {
  FeatureMatrix const x = MAT(Feature, rows(4), 0.0F, 0.0F, 0.1F, 0.1F, 9.9F, 0.0F, 9.8F, 0.1F);

  OutcomeVector const y = VEC(Outcome, 0, 1, 1, 1);

  auto condition =
      TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F);

  Forest forest;
  forest.add_tree(
      std::make_unique<BootstrapTree>(std::move(condition), TrainingSpec::builder().make(), std::vector<int>{0, 2})
  );

  double const err = forest.oob_error(x, y);

  ASSERT_DOUBLE_EQ(err, 0.5);
}

// ---------------------------------------------------------------------------
// oob_predict
// ---------------------------------------------------------------------------

TEST(OobPredict, HandBuiltTreeWithKnownOob) {
  FeatureMatrix const x = MAT(Feature, rows(6), 0.0F, 0.5F, 0.1F, 0.3F, 0.2F, 0.7F, 9.8F, 0.4F, 9.9F, 0.6F, 9.7F, 0.2F);

  auto condition =
      TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F);

  Forest forest;
  forest.add_tree(std::make_unique<BootstrapTree>(
      std::move(condition), TrainingSpec::builder().make(), std::vector<int>{0, 1, 4, 5}
  ));

  OutcomeVector preds = forest.oob_predict(x);

  ASSERT_EQ(preds.size(), 6);
  EXPECT_EQ(preds(0), -1) << "Row 0 in bag";
  EXPECT_EQ(preds(1), -1) << "Row 1 in bag";
  EXPECT_EQ(preds(2), 0) << "Row 2 OOB, x[0]=0.2 < 5";
  EXPECT_EQ(preds(3), 1) << "Row 3 OOB, x[0]=9.8 > 5";
  EXPECT_EQ(preds(4), -1) << "Row 4 in bag";
  EXPECT_EQ(preds(5), -1) << "Row 5 in bag";
}

TEST(OobPredict, AllInBagReturnsSentinel) {
  FeatureMatrix const x = MAT(Feature, rows(4), 0.0F, 0.0F, 0.1F, 0.1F, 9.9F, 0.0F, 9.8F, 0.1F);

  auto condition =
      TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F);

  Forest forest;
  forest.add_tree(std::make_unique<BootstrapTree>(
      std::move(condition), TrainingSpec::builder().make(), std::vector<int>{0, 1, 2, 3}
  ));

  OutcomeVector const preds = forest.oob_predict(x);

  ASSERT_EQ(preds.size(), 4);
  EXPECT_EQ(preds(0), -1);
  EXPECT_EQ(preds(1), -1);
  EXPECT_EQ(preds(2), -1);
  EXPECT_EQ(preds(3), -1);
}

TEST(OobPredict, HandBuiltTreeWithOobMisclassification) {
  FeatureMatrix const x = MAT(Feature, rows(4), 0.0F, 0.0F, 0.1F, 0.1F, 9.9F, 0.0F, 9.8F, 0.1F);

  auto condition =
      TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F);

  Forest forest;
  forest.add_tree(
      std::make_unique<BootstrapTree>(std::move(condition), TrainingSpec::builder().make(), std::vector<int>{0, 2})
  );

  OutcomeVector const preds = forest.oob_predict(x);

  ASSERT_EQ(preds.size(), 4);
  EXPECT_EQ(preds(0), -1) << "Row 0 in bag";
  EXPECT_EQ(preds(1), 0) << "Row 1 OOB, x[0]=0.1 < 5 -> group 0";
  EXPECT_EQ(preds(2), -1) << "Row 2 in bag";
  EXPECT_EQ(preds(3), 1) << "Row 3 OOB, x[0]=9.8 > 5 -> group 1";
}

TEST(OobPredict, ConsistentWithOobError) {
  FeatureMatrix const x =
      MAT(Feature,
          rows(20),
          0.0F,
          1.0F,
          0.1F,
          2.0F,
          0.2F,
          0.5F,
          0.3F,
          1.5F,
          0.4F,
          0.8F,
          0.5F,
          1.2F,
          0.6F,
          0.9F,
          0.7F,
          1.8F,
          0.8F,
          0.6F,
          0.9F,
          1.1F,
          9.0F,
          1.0F,
          9.1F,
          2.0F,
          9.2F,
          0.5F,
          9.3F,
          1.5F,
          9.4F,
          0.8F,
          9.5F,
          1.2F,
          9.6F,
          0.9F,
          9.7F,
          1.8F,
          9.8F,
          0.6F,
          9.9F,
          1.1F);

  OutcomeVector y = VEC(Outcome, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

  Forest const forest = Forest::train(TrainingSpec::builder().size(50).build(), x, y);

  OutcomeVector const preds = forest.oob_predict(x);
  double const err          = forest.oob_error(x, y);

  int evaluated = 0;
  int correct   = 0;

  for (int i = 0; i < preds.size(); ++i) {
    if (preds(i) >= 0) {
      ++evaluated;

      if (preds(i) == y(i)) {
        ++correct;
      }
    }
  }

  double const expected_err =
      (evaluated == 0) ? -1.0 : 1.0 - static_cast<double>(correct) / static_cast<double>(evaluated);

  ASSERT_DOUBLE_EQ(err, expected_err) << "oob_error should match manual computation from oob_predict";
}

// ---------------------------------------------------------------------------
// Edge cases — "doesn't blow up" tests
// ---------------------------------------------------------------------------

TEST(ForestEdgeCase, ConstantFeatureColumn) {
  FeatureMatrix const x = MAT(Feature, rows(6), 5, 1, 5, 2, 5, 3, 5, 7, 5, 8, 5, 9);
  OutcomeVector const y = VEC(Outcome, 0, 0, 0, 1, 1, 1);

  Forest const forest = Forest::train(TrainingSpec::builder().size(5).threads(1).build(), x, y);

  OutcomeVector const predictions = forest.predict(x);
  ASSERT_EQ(predictions.size(), y.size());
  EXPECT_TRUE((predictions.array() >= 0).all() && (predictions.array() <= 1).all());
}

TEST(ForestEdgeCase, SingleObservationPerGroup) {
  FeatureMatrix const x = MAT(Feature, rows(2), 1, 0, 0, 1);
  OutcomeVector const y = VEC(Outcome, 0, 1);

  Forest const forest = Forest::train(TrainingSpec::builder().size(5).threads(1).build(), x, y);

  OutcomeVector const predictions = forest.predict(x);
  ASSERT_EQ(predictions.size(), y.size());
  EXPECT_TRUE((predictions.array() >= 0).all() && (predictions.array() <= 1).all());
}

TEST(ForestEdgeCase, MinimalDataset) {
  FeatureMatrix const x = MAT(Feature, rows(2), 1, 9);
  OutcomeVector const y = VEC(Outcome, 0, 1);

  Forest const forest = Forest::train(TrainingSpec::builder().size(5).threads(1).build(), x, y);

  OutcomeVector const predictions = forest.predict(x);
  ASSERT_EQ(predictions.size(), y.size());
  EXPECT_TRUE((predictions.array() >= 0).all() && (predictions.array() <= 1).all());
}

TEST(ForestEdgeCase, ExtremeImbalance) {
  // clang-format off
  FeatureMatrix const x = MAT(Feature, rows(20),
    0, 0,  1, 1,  2, 2,  3, 0,  4, 1,
    0, 2,  1, 0,  2, 1,  3, 2,  4, 0,
    0, 1,  1, 2,  2, 0,  3, 1,  4, 2,
    0, 0,  1, 1,  2, 2,  90, 90,  91, 91);
  OutcomeVector const y = VEC(Outcome,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1);
  // clang-format on

  Forest const forest = Forest::train(TrainingSpec::builder().size(10).threads(1).build(), x, y);

  OutcomeVector const predictions = forest.predict(x);
  ASSERT_EQ(predictions.size(), y.size());
  EXPECT_TRUE((predictions.array() >= 0).all() && (predictions.array() <= 1).all());
}
