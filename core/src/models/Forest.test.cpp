#include <gtest/gtest.h>

#include "models/BootstrapTree.hpp"
#include "models/Forest.hpp"
#include "models/TreeCondition.hpp"
#include "models/TreeResponse.hpp"

#include "models/TrainingSpec.hpp"

#include "stats/Simulation.hpp"
#include "stats/Stats.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::stats;
using namespace ppforest2::types;
using namespace ppforest2::pp;
using namespace ppforest2::math;

static Projector as_projector(std::vector<Feature> vector) {
  Eigen::Map<Projector> projector(vector.data(), vector.size());
  return projector;
}

TEST(Forest, TrainLDAAllVariablesProperties) {
  FeatureMatrix x =
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

  ResponseVector y =
      VEC(Response, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);

  int const n_vars   = x.cols();
  float const lambda = 0;
  int const seed     = 0;

  Forest result = Forest::train(TrainingSpec(pp::pda(lambda), dr::uniform(n_vars), sr::mean_of_means(), 4, seed), x, y);

  ASSERT_EQ(result.trees.size(), 4);
  ASSERT_EQ(result.training_spec->seed, seed);

  ResponseVector predictions = result.predict(x);
  double err                 = error_rate(predictions, y);

  ASSERT_EQ(err, 0.0) << "Forest should achieve 0% training error on well-separated 3-group data";
}

TEST(Forest, TrainLDASomeVariablesProperties) {
  FeatureMatrix x =
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

  ResponseVector y =
      VEC(Response, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);

  int const n_vars   = 2;
  float const lambda = 0;
  int const seed     = 1;


  Forest result = Forest::train(TrainingSpec(pp::pda(lambda), dr::uniform(n_vars), sr::mean_of_means(), 4, seed), x, y);

  ASSERT_EQ(result.trees.size(), 4);
  ASSERT_EQ(result.training_spec->seed, seed);

  ResponseVector predictions = result.predict(x);
  double err                 = error_rate(predictions, y);

  ASSERT_LT(err, 0.30) << "Forest with subset of variables should still classify well-separated data";
}

TEST(Forest, TrainPDAAllVariablesProperties) {
  FeatureMatrix x =
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

  ResponseVector y = VEC(Response, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

  int const n_vars   = x.cols();
  float const lambda = 0.1;
  int const seed     = 0;

  Forest result = Forest::train(TrainingSpec(pp::pda(lambda), dr::uniform(n_vars), sr::mean_of_means(), 4, seed), x, y);

  ASSERT_EQ(result.trees.size(), 4);
  ASSERT_EQ(result.training_spec->seed, seed);

  ResponseVector predictions = result.predict(x);
  double err                 = error_rate(predictions, y);

  ASSERT_EQ(err, 0.0) << "PDA forest should achieve 0% training error on well-separated 2-group data";
}

TEST(ForestSimulation, PerfectSeparationLowOOBError) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 200.0f;
  params.sd              = 1.0f;

  auto data = simulate(90, 4, 3, rng, params);

  Forest forest =
      Forest::train(TrainingSpec(pp::pda(0.0f), dr::uniform(4), sr::mean_of_means(), 20, 0, 1), data.x, data.y);

  double err = forest.oob_error(data.x, data.y);

  ASSERT_GE(err, 0.0);
  ASSERT_LE(err, 0.05) << "Forest should achieve near-zero OOB error on perfectly separated data";
}

TEST(ForestSimulation, HighOverlapBoundedError) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 5.0f;
  params.sd              = 50.0f;

  auto data = simulate(200, 4, 3, rng, params);

  Forest forest =
      Forest::train(TrainingSpec(pp::pda(0.0f), dr::uniform(4), sr::mean_of_means(), 20, 0, 1), data.x, data.y);

  ResponseVector predictions = forest.predict(data.x);
  double err                 = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.80) << "Forest error on highly overlapping data should be bounded";
}

TEST(ForestSimulation, ManyClasses) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 50.0f;
  params.sd              = 10.0f;

  auto data = simulate(300, 4, 10, rng, params);

  Forest forest =
      Forest::train(TrainingSpec(pp::pda(0.0f), dr::uniform(4), sr::mean_of_means(), 20, 0, 1), data.x, data.y);

  ResponseVector predictions = forest.predict(data.x);
  double err                 = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.30) << "Forest should handle 10 groups with reasonable error";
}

TEST(ForestSimulation, HighDimensionality) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 50.0f;
  params.sd              = 10.0f;

  auto data = simulate(100, 50, 3, rng, params);

  Forest forest =
      Forest::train(TrainingSpec(pp::pda(0.0f), dr::uniform(7), sr::mean_of_means(), 20, 0, 1), data.x, data.y);

  ResponseVector predictions = forest.predict(data.x);
  double err                 = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.20) << "Forest should handle high-dimensional data (p=50)";
}

TEST(ForestSimulation, Deterministic) {
  RNG rng(0);
  auto data = simulate(90, 4, 3, rng);

  Forest f1 = Forest::train(TrainingSpec(pp::pda(0.0f), dr::uniform(4), sr::mean_of_means(), 10, 0, 1), data.x, data.y);
  Forest f2 = Forest::train(TrainingSpec(pp::pda(0.0f), dr::uniform(4), sr::mean_of_means(), 10, 0, 1), data.x, data.y);

  ASSERT_EQ(f1, f2) << "Same seed should produce identical forests";
}

TEST(ForestSimulation, PDAOnOverlappingData) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 10.0f;
  params.sd              = 20.0f;

  auto data = simulate(200, 4, 3, rng, params);

  Forest forest =
      Forest::train(TrainingSpec(pp::pda(0.5f), dr::uniform(4), sr::mean_of_means(), 20, 0, 1), data.x, data.y);

  ResponseVector predictions = forest.predict(data.x);
  double err                 = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.80) << "PDA forest should produce bounded error on noisy data";
}

TEST(ForestSimulation, LargeDataset) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 50.0f;
  params.sd              = 10.0f;

  auto data = simulate(2000, 10, 4, rng, params);

  Forest forest =
      Forest::train(TrainingSpec(pp::pda(0.0f), dr::uniform(5), sr::mean_of_means(), 10, 0, 1), data.x, data.y);

  ResponseVector predictions = forest.predict(data.x);
  double err                 = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.10) << "Forest should handle large datasets efficiently";
}

static Forest build_three_group_forest() {
  Forest forest;

  forest.add_tree(std::make_unique<BootstrapTree>(
      TreeCondition::make(
          as_projector({0.0, 0.0, 0.0, 0.598, -0.801}),
          -0.348,
          TreeCondition::make(
              as_projector({0.9995, 0.0, -0.031, 0.0, 0.0}), 5.553, TreeResponse::make(1), TreeResponse::make(2)
          ),
          TreeResponse::make(0)
      ),
      nullptr
  ));

  forest.add_tree(std::make_unique<BootstrapTree>(
      TreeCondition::make(
          as_projector({0.9998, 0.0, -0.019, 0.0, 0.0}),
          5.300,
          TreeCondition::make(
              as_projector({0.999, 0.0, 0.0, 0.046, 0.0}), 1.609, TreeResponse::make(0), TreeResponse::make(1)
          ),
          TreeResponse::make(2)
      ),
      nullptr
  ));

  forest.add_tree(std::make_unique<BootstrapTree>(
      TreeCondition::make(
          as_projector({0.974, -0.226, 0.0, 0.0, 0.0}),
          3.955,
          TreeCondition::make(
              as_projector({0.0, 0.9996, -0.030, 0.0, 0.0}), 2.622, TreeResponse::make(0), TreeResponse::make(1)
          ),
          TreeResponse::make(2)
      ),
      nullptr
  ));

  forest.add_tree(std::make_unique<BootstrapTree>(
      TreeCondition::make(
          as_projector({0.962, 0.0, 0.0, 0.0, -0.275}),
          4.735,
          TreeCondition::make(
              as_projector({0.0, 0.0, 0.377, 0.0, -0.926}), -0.832, TreeResponse::make(1), TreeResponse::make(0)
          ),
          TreeResponse::make(2)
      ),
      nullptr
  ));

  return forest;
}

TEST(Forest, PredictSingleObservation) {
  Forest forest = build_three_group_forest();

  ASSERT_EQ(0, forest.predict(VEC(Feature, 1, 0, 1, 1, 1)));
  ASSERT_EQ(1, forest.predict(VEC(Feature, 2, 5, 0, 0, 1)));
  ASSERT_EQ(2, forest.predict(VEC(Feature, 9, 8, 1, 1, 1)));
}

TEST(Forest, PredictBatch) {
  Forest forest = build_three_group_forest();

  FeatureMatrix x =
      MAT(Feature, rows(6), 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 2, 5, 0, 0, 1, 2, 5, 1, 0, 1, 9, 8, 0, 0, 1, 9, 8, 1, 0, 2);

  ResponseVector result   = forest.predict(x);
  ResponseVector expected = VEC(Response, 0, 0, 1, 1, 2, 2);

  ASSERT_EQ(expected, result);
}

// ---------------------------------------------------------------------------
// OOB error
// ---------------------------------------------------------------------------

TEST(OobError, PerfectSeparationGivesLowError) {
  FeatureMatrix x =
      MAT(Feature,
          rows(20),
          0.0f,
          1.0f,
          0.1f,
          2.0f,
          0.2f,
          0.5f,
          0.3f,
          1.5f,
          0.4f,
          0.8f,
          0.5f,
          1.2f,
          0.6f,
          0.9f,
          0.7f,
          1.8f,
          0.8f,
          0.6f,
          0.9f,
          1.1f,
          9.0f,
          1.0f,
          9.1f,
          2.0f,
          9.2f,
          0.5f,
          9.3f,
          1.5f,
          9.4f,
          0.8f,
          9.5f,
          1.2f,
          9.6f,
          0.9f,
          9.7f,
          1.8f,
          9.8f,
          0.6f,
          9.9f,
          1.1f);

  ResponseVector y = VEC(Response, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

  Forest forest = Forest::train(TrainingSpec(pp::pda(0.0f), dr::noop(), sr::mean_of_means(), 50, 0), x, y);

  double err = forest.oob_error(x, y);

  ASSERT_GE(err, 0.0);
  ASSERT_LE(err, 0.1) << "Expected near-zero OOB error for perfectly separable data";
}

TEST(OobError, AllInBagReturnsNegative) {
  FeatureMatrix x = MAT(Feature, rows(4), 0.0f, 0.0f, 0.1f, 0.1f, 9.9f, 0.0f, 9.8f, 0.1f);

  ResponseVector y = VEC(Response, 0, 0, 1, 1);

  auto condition =
      TreeCondition::make(as_projector({1.0f, 0.0f}), 5.0f, TreeResponse::make(0), TreeResponse::make(1), {0, 1}, 0.9f);

  Forest forest;
  forest.add_tree(std::make_unique<BootstrapTree>(
      std::move(condition),
      TrainingSpec::make(pp::pda(0.0f), dr::noop(), sr::mean_of_means()),
      std::vector<int>{0, 1, 2, 3}
  ));

  double err = forest.oob_error(x, y);

  ASSERT_DOUBLE_EQ(err, -1.0) << "No OOB observations, should return -1";
}

TEST(OobError, HandBuiltTreeWithKnownOob) {
  FeatureMatrix x = MAT(Feature, rows(6), 0.0f, 0.5f, 0.1f, 0.3f, 0.2f, 0.7f, 9.8f, 0.4f, 9.9f, 0.6f, 9.7f, 0.2f);

  ResponseVector y = VEC(Response, 0, 0, 0, 1, 1, 1);

  auto condition =
      TreeCondition::make(as_projector({1.0f, 0.0f}), 5.0f, TreeResponse::make(0), TreeResponse::make(1), {0, 1}, 0.9f);

  Forest forest;
  forest.add_tree(std::make_unique<BootstrapTree>(
      std::move(condition),
      TrainingSpec::make(pp::pda(0.0f), dr::noop(), sr::mean_of_means()),
      std::vector<int>{0, 1, 4, 5}
  ));

  double err = forest.oob_error(x, y);

  ASSERT_DOUBLE_EQ(err, 0.0);
}

TEST(OobError, HandBuiltTreeWithOobMisclassification) {
  FeatureMatrix x = MAT(Feature, rows(4), 0.0f, 0.0f, 0.1f, 0.1f, 9.9f, 0.0f, 9.8f, 0.1f);

  ResponseVector y = VEC(Response, 0, 1, 1, 1);

  auto condition =
      TreeCondition::make(as_projector({1.0f, 0.0f}), 5.0f, TreeResponse::make(0), TreeResponse::make(1), {0, 1}, 0.9f);

  Forest forest;
  forest.add_tree(std::make_unique<BootstrapTree>(
      std::move(condition), TrainingSpec::make(pp::pda(0.0f), dr::noop(), sr::mean_of_means()), std::vector<int>{0, 2}
  ));

  double err = forest.oob_error(x, y);

  ASSERT_DOUBLE_EQ(err, 0.5);
}

// ---------------------------------------------------------------------------
// oob_predict
// ---------------------------------------------------------------------------

TEST(OobPredict, HandBuiltTreeWithKnownOob) {
  FeatureMatrix x = MAT(Feature, rows(6), 0.0f, 0.5f, 0.1f, 0.3f, 0.2f, 0.7f, 9.8f, 0.4f, 9.9f, 0.6f, 9.7f, 0.2f);

  auto condition =
      TreeCondition::make(as_projector({1.0f, 0.0f}), 5.0f, TreeResponse::make(0), TreeResponse::make(1), {0, 1}, 0.9f);

  Forest forest;
  forest.add_tree(std::make_unique<BootstrapTree>(
      std::move(condition),
      TrainingSpec::make(pp::pda(0.0f), dr::noop(), sr::mean_of_means()),
      std::vector<int>{0, 1, 4, 5}
  ));

  ResponseVector preds = forest.oob_predict(x);

  ASSERT_EQ(preds.size(), 6);
  EXPECT_EQ(preds(0), -1) << "Row 0 in bag";
  EXPECT_EQ(preds(1), -1) << "Row 1 in bag";
  EXPECT_EQ(preds(2), 0) << "Row 2 OOB, x[0]=0.2 < 5";
  EXPECT_EQ(preds(3), 1) << "Row 3 OOB, x[0]=9.8 > 5";
  EXPECT_EQ(preds(4), -1) << "Row 4 in bag";
  EXPECT_EQ(preds(5), -1) << "Row 5 in bag";
}

TEST(OobPredict, AllInBagReturnsSentinel) {
  FeatureMatrix x = MAT(Feature, rows(4), 0.0f, 0.0f, 0.1f, 0.1f, 9.9f, 0.0f, 9.8f, 0.1f);

  auto condition =
      TreeCondition::make(as_projector({1.0f, 0.0f}), 5.0f, TreeResponse::make(0), TreeResponse::make(1), {0, 1}, 0.9f);

  Forest forest;
  forest.add_tree(std::make_unique<BootstrapTree>(
      std::move(condition),
      TrainingSpec::make(pp::pda(0.0f), dr::noop(), sr::mean_of_means()),
      std::vector<int>{0, 1, 2, 3}
  ));

  ResponseVector preds = forest.oob_predict(x);

  ASSERT_EQ(preds.size(), 4);
  EXPECT_EQ(preds(0), -1);
  EXPECT_EQ(preds(1), -1);
  EXPECT_EQ(preds(2), -1);
  EXPECT_EQ(preds(3), -1);
}

TEST(OobPredict, HandBuiltTreeWithOobMisclassification) {
  FeatureMatrix x = MAT(Feature, rows(4), 0.0f, 0.0f, 0.1f, 0.1f, 9.9f, 0.0f, 9.8f, 0.1f);

  auto condition =
      TreeCondition::make(as_projector({1.0f, 0.0f}), 5.0f, TreeResponse::make(0), TreeResponse::make(1), {0, 1}, 0.9f);

  Forest forest;
  forest.add_tree(std::make_unique<BootstrapTree>(
      std::move(condition), TrainingSpec::make(pp::pda(0.0f), dr::noop(), sr::mean_of_means()), std::vector<int>{0, 2}
  ));

  ResponseVector preds = forest.oob_predict(x);

  ASSERT_EQ(preds.size(), 4);
  EXPECT_EQ(preds(0), -1) << "Row 0 in bag";
  EXPECT_EQ(preds(1), 0) << "Row 1 OOB, x[0]=0.1 < 5 -> group 0";
  EXPECT_EQ(preds(2), -1) << "Row 2 in bag";
  EXPECT_EQ(preds(3), 1) << "Row 3 OOB, x[0]=9.8 > 5 -> group 1";
}

TEST(OobPredict, ConsistentWithOobError) {
  FeatureMatrix x =
      MAT(Feature,
          rows(20),
          0.0f,
          1.0f,
          0.1f,
          2.0f,
          0.2f,
          0.5f,
          0.3f,
          1.5f,
          0.4f,
          0.8f,
          0.5f,
          1.2f,
          0.6f,
          0.9f,
          0.7f,
          1.8f,
          0.8f,
          0.6f,
          0.9f,
          1.1f,
          9.0f,
          1.0f,
          9.1f,
          2.0f,
          9.2f,
          0.5f,
          9.3f,
          1.5f,
          9.4f,
          0.8f,
          9.5f,
          1.2f,
          9.6f,
          0.9f,
          9.7f,
          1.8f,
          9.8f,
          0.6f,
          9.9f,
          1.1f);

  ResponseVector y = VEC(Response, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

  Forest forest = Forest::train(TrainingSpec(pp::pda(0.0f), dr::noop(), sr::mean_of_means(), 50, 0), x, y);

  ResponseVector preds = forest.oob_predict(x);
  double err           = forest.oob_error(x, y);

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

  double expected_err = (evaluated == 0) ? -1.0 : 1.0 - static_cast<double>(correct) / static_cast<double>(evaluated);

  ASSERT_DOUBLE_EQ(err, expected_err) << "oob_error should match manual computation from oob_predict";
}
