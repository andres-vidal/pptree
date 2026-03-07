#include <gtest/gtest.h>

#include "models/BootstrapTree.hpp"
#include "models/Forest.hpp"
#include "models/TreeCondition.hpp"
#include "models/TreeResponse.hpp"

#include "models/TrainingSpec.hpp"
#include "models/TrainingSpecGLDA.hpp"
#include "models/TrainingSpecUGLDA.hpp"

#include "stats/Simulation.hpp"
#include "stats/Stats.hpp"
#include "utils/Macros.hpp"

using namespace pptree;
using namespace pptree::stats;
using namespace pptree::types;
using namespace pptree::pp;
using namespace pptree::math;

static Projector as_projector(std::vector<Feature> vector) {
  Eigen::Map<Projector > projector(vector.data(), vector.size());
  return projector;
}

TEST(Forest, TrainLDAAllVariablesProperties) {
  FeatureMatrix x = DATA(Feature, 30,
      1, 0, 1, 1, 1,
      1, 0, 1, 0, 0,
      1, 0, 0, 0, 1,
      1, 0, 1, 2, 1,
      1, 0, 0, 1, 1,
      1, 1, 1, 1, 0,
      1, 0, 0, 2, 1,
      1, 0, 1, 1, 2,
      1, 0, 0, 2, 0,
      1, 0, 2, 1, 0,
      2, 5, 0, 0, 1,
      2, 5, 0, 0, 2,
      3, 5, 1, 0, 2,
      2, 5, 1, 0, 1,
      2, 5, 0, 1, 1,
      2, 5, 0, 1, 2,
      2, 5, 2, 1, 1,
      2, 5, 1, 1, 1,
      2, 5, 1, 1, 2,
      2, 5, 2, 1, 2,
      2, 5, 1, 2, 1,
      2, 5, 2, 1, 1,
      9, 8, 0, 0, 1,
      9, 8, 0, 0, 2,
      9, 8, 1, 0, 2,
      9, 8, 1, 0, 1,
      9, 8, 0, 1, 1,
      9, 8, 0, 1, 2,
      9, 8, 2, 1, 1,
      9, 8, 1, 1, 1);

  ResponseVector y = DATA(Response, 30,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      2, 2, 2, 2, 2, 2, 2, 2);

  const int n_vars   = x.cols();
  const float lambda = 0;
  const int seed     = 0;

  Forest result = Forest::train(
    TrainingSpecUGLDA(n_vars, lambda),
    x, y, 4, seed);

  ASSERT_EQ(result.trees.size(), 4);
  ASSERT_EQ(result.seed, seed);

  ResponseVector predictions = result.predict(x);
  double err                 = error_rate(predictions, y);

  ASSERT_EQ(err, 0.0) << "Forest should achieve 0% training error on well-separated 3-group data";
}

TEST(Forest, TrainLDASomeVariablesProperties) {
  FeatureMatrix x = DATA(Feature, 30,
      1, 0, 1, 1, 1,
      1, 0, 1, 0, 0,
      1, 0, 0, 0, 1,
      1, 0, 1, 2, 1,
      1, 0, 0, 1, 1,
      1, 1, 1, 1, 0,
      1, 0, 0, 2, 1,
      1, 0, 1, 1, 2,
      1, 0, 0, 2, 0,
      1, 0, 2, 1, 0,
      2, 5, 0, 0, 1,
      2, 5, 0, 0, 2,
      3, 5, 1, 0, 2,
      2, 5, 1, 0, 1,
      2, 5, 0, 1, 1,
      2, 5, 0, 1, 2,
      2, 5, 2, 1, 1,
      2, 5, 1, 1, 1,
      2, 5, 1, 1, 2,
      2, 5, 2, 1, 2,
      2, 5, 1, 2, 1,
      2, 5, 2, 1, 1,
      9, 8, 0, 0, 1,
      9, 8, 0, 0, 2,
      9, 8, 1, 0, 2,
      9, 8, 1, 0, 1,
      9, 8, 0, 1, 1,
      9, 8, 0, 1, 2,
      9, 8, 2, 1, 1,
      9, 8, 1, 1, 1);

  ResponseVector y = DATA(Response, 30,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      2, 2, 2, 2, 2, 2, 2, 2);

  const int n_vars   = 2;
  const float lambda = 0;
  const int seed     = 1;

  Forest result = Forest::train(
    TrainingSpecUGLDA(n_vars, lambda),
    x, y, 4, seed);

  ASSERT_EQ(result.trees.size(), 4);
  ASSERT_EQ(result.seed, seed);

  ResponseVector predictions = result.predict(x);
  double err                 = error_rate(predictions, y);

  ASSERT_LT(err, 0.30) << "Forest with subset of variables should still classify well-separated data";
}

TEST(Forest, TrainPDAAllVariablesProperties) {
  FeatureMatrix x = DATA(Feature, 10,
      1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      4, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2,
      5, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      4, 0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 2,
      4, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2,
      4, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
      4, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2);

  ResponseVector y = DATA(Response, 10,
      0, 0, 0, 0,
      1, 1, 1, 1, 1, 1);

  const int n_vars   = x.cols();
  const float lambda = 0.1;
  const int seed     = 0;

  Forest result = Forest::train(
    TrainingSpecUGLDA(n_vars, lambda),
    x, y, 4, seed);

  ASSERT_EQ(result.trees.size(), 4);
  ASSERT_EQ(result.seed, seed);

  ResponseVector predictions = result.predict(x);
  double err                 = error_rate(predictions, y);

  ASSERT_EQ(err, 0.0) << "PDA forest should achieve 0% training error on well-separated 2-group data";
}

TEST(ForestSimulation, PerfectSeparationLowOOBError) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 200.0f;
  params.sd              = 1.0f;

  auto data = simulate(90, 4, 3, rng, params);

  Forest forest = Forest::train(TrainingSpecUGLDA(4, 0.0f), data.x, data.y, 20, 42, 1);

  double err = forest.oob_error(data.x, data.y);

  ASSERT_GE(err, 0.0);
  ASSERT_LE(err, 0.05) << "Forest should achieve near-zero OOB error on perfectly separated data";
}

TEST(ForestSimulation, HighOverlapBoundedError) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 5.0f;
  params.sd              = 50.0f;

  auto data = simulate(200, 4, 3, rng, params);

  Forest forest = Forest::train(TrainingSpecUGLDA(4, 0.0f), data.x, data.y, 20, 42, 1);

  ResponseVector predictions = forest.predict(data.x);
  double err                 = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.80) << "Forest error on highly overlapping data should be bounded";
}

TEST(ForestSimulation, ManyClasses) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 50.0f;
  params.sd              = 10.0f;

  auto data = simulate(300, 4, 10, rng, params);

  Forest forest = Forest::train(TrainingSpecUGLDA(4, 0.0f), data.x, data.y, 20, 42, 1);

  ResponseVector predictions = forest.predict(data.x);
  double err                 = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.30) << "Forest should handle 10 classes with reasonable error";
}

TEST(ForestSimulation, HighDimensionality) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 50.0f;
  params.sd              = 10.0f;

  auto data = simulate(100, 50, 3, rng, params);

  Forest forest = Forest::train(TrainingSpecUGLDA(7, 0.0f), data.x, data.y, 20, 42, 1);

  ResponseVector predictions = forest.predict(data.x);
  double err                 = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.20) << "Forest should handle high-dimensional data (p=50)";
}

TEST(ForestSimulation, Deterministic) {
  RNG rng(42);
  auto data = simulate(90, 4, 3, rng);

  Forest f1 = Forest::train(TrainingSpecUGLDA(4, 0.0f), data.x, data.y, 10, 42, 1);
  Forest f2 = Forest::train(TrainingSpecUGLDA(4, 0.0f), data.x, data.y, 10, 42, 1);

  ASSERT_EQ(f1, f2) << "Same seed should produce identical forests";
}

TEST(ForestSimulation, PDAOnOverlappingData) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 10.0f;
  params.sd              = 20.0f;

  auto data = simulate(200, 4, 3, rng, params);

  Forest forest = Forest::train(TrainingSpecUGLDA(4, 0.5f), data.x, data.y, 20, 42, 1);

  ResponseVector predictions = forest.predict(data.x);
  double err                 = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.80) << "PDA forest should produce bounded error on noisy data";
}

TEST(ForestSimulation, LargeDataset) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 50.0f;
  params.sd              = 10.0f;

  auto data = simulate(2000, 10, 4, rng, params);

  Forest forest = Forest::train(TrainingSpecUGLDA(5, 0.0f), data.x, data.y, 10, 42, 1);

  ResponseVector predictions = forest.predict(data.x);
  double err                 = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.10) << "Forest should handle large datasets efficiently";
}

TEST(Forest, PredictDataColumnSomeVariablesMultivariateThreeGroups) {
  Forest forest;

  forest.add_tree(
    std::make_unique<BootstrapTree>(
      TreeCondition::make(
        as_projector({ 0.0, 0.0, 0.0, 0.5982325379690726, -0.8013225508589422 }),
        -0.3483987096124312,
        TreeCondition::make(
          as_projector({ 0.999534397402818, 0.0, -0.030512102657559676, 0.0, 0.0 }),
          5.55339996020167,
          TreeResponse::make(1),
          TreeResponse::make(2)),
        TreeResponse::make(0)
        ))
    );


  forest.add_tree(
    std::make_unique<BootstrapTree>(
      TreeCondition::make(
        as_projector({ 0.9998222455113714, 0.0, -0.018854107791118468, 0.0, 0.0 }),
        5.300417766337716,
        TreeCondition::make(
          as_projector({ 0.9989543519613864, 0.0, 0.0, 0.0457187346435417, 0.0 }),
          1.6094899541803496,
          TreeResponse::make(0),
          TreeResponse::make(1)),
        TreeResponse::make(2)))
    );


  forest.add_tree(
    std::make_unique<BootstrapTree>(
      TreeCondition::make(
        as_projector({ 0.9741975531020036, -0.2256969816591904, 0.0, 0.0, 0.0 }),
        3.9550147456664178,
        TreeCondition::make(
          as_projector({ 0.0, 0.9995561292785718, -0.029791683766431428, 0.0, 0.0 }),
          2.6217629631670403,
          TreeResponse::make(0),
          TreeResponse::make(1)),
        TreeResponse::make(2)))
    );


  forest.add_tree(
    std::make_unique<BootstrapTree>(
      TreeCondition::make(
        as_projector({ 0.9615748657636985, 0.0, 0.0, 0.0, -0.2745428519038971 }),
        4.734758305714628,
        TreeCondition::make(
          as_projector({ 0.0, 0.0, 0.3772334858435029, 0.0, -0.926118187467647 }),
          -0.8315603229605784,
          TreeResponse::make(1),
          TreeResponse::make(0)),
        TreeResponse::make(2)))
    );

  FeatureVector x = DATA(Feature, 5, 9, 8, 1, 1, 1);

  int result = forest.predict(x);

  ASSERT_EQ(2, result);
}

TEST(Forest, PredictDataSomeVariablesMultivariateThreeGroups) {
  Forest forest;

  forest.add_tree(
    std::make_unique<BootstrapTree>(
      TreeCondition::make(
        as_projector({ 0.0, 0.0, 0.0, 0.5982325379690726, -0.8013225508589422 }),
        -0.3483987096124312,
        TreeCondition::make(
          as_projector({ 0.999534397402818, 0.0, -0.030512102657559676, 0.0, 0.0 }),
          5.55339996020167,
          TreeResponse::make(1),
          TreeResponse::make(2)),
        TreeResponse::make(0))
      )
    );


  forest.add_tree(
    std::make_unique<BootstrapTree>(
      TreeCondition::make(
        as_projector({ 0.9998222455113714, 0.0, -0.018854107791118468, 0.0, 0.0 }),
        5.300417766337716,
        TreeCondition::make(
          as_projector({ 0.9989543519613864, 0.0, 0.0, 0.0457187346435417, 0.0 }),
          1.6094899541803496,
          TreeResponse::make(0),
          TreeResponse::make(1)),
        TreeResponse::make(2))
      )
    );


  forest.add_tree(
    std::make_unique<BootstrapTree>(
      TreeCondition::make(
        as_projector({ 0.9741975531020036, -0.2256969816591904, 0.0, 0.0, 0.0 }),
        3.9550147456664178,
        TreeCondition::make(
          as_projector({ 0.0, 0.9995561292785718, -0.029791683766431428, 0.0, 0.0 }),
          2.6217629631670403,
          TreeResponse::make(0),
          TreeResponse::make(1)),
        TreeResponse::make(2))
      )
    );


  forest.add_tree(
    std::make_unique<BootstrapTree>(
      TreeCondition::make(
        as_projector({ 0.9615748657636985, 0.0, 0.0, 0.0, -0.2745428519038971 }),
        4.734758305714628,
        TreeCondition::make(
          as_projector({ 0.0, 0.0, 0.3772334858435029, 0.0, -0.926118187467647 }),
          -0.8315603229605784,
          TreeResponse::make(1),
          TreeResponse::make(0)),
        TreeResponse::make(2))
      )
    );

  FeatureMatrix x = DATA(Feature, 30,
      1, 0, 1, 1, 1,
      1, 0, 1, 0, 0,
      1, 0, 0, 0, 1,
      1, 0, 1, 2, 1,
      1, 0, 0, 1, 1,
      1, 1, 1, 1, 0,
      1, 0, 0, 2, 1,
      1, 0, 1, 1, 2,
      1, 0, 0, 2, 0,
      1, 0, 2, 1, 0,
      2, 5, 0, 0, 1,
      2, 5, 0, 0, 2,
      3, 5, 1, 0, 2,
      2, 5, 1, 0, 1,
      2, 5, 0, 1, 1,
      2, 5, 0, 1, 2,
      2, 5, 2, 1, 1,
      2, 5, 1, 1, 1,
      2, 5, 1, 1, 2,
      2, 5, 2, 1, 2,
      2, 5, 1, 2, 1,
      2, 5, 2, 1, 1,
      9, 8, 0, 0, 1,
      9, 8, 0, 0, 2,
      9, 8, 1, 0, 2,
      9, 8, 1, 0, 1,
      9, 8, 0, 1, 1,
      9, 8, 0, 1, 2,
      9, 8, 2, 1, 1,
      9, 8, 1, 1, 1);

  ResponseVector y = DATA(Response, 30,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
      2, 2, 2, 2, 2, 2, 2, 2);

  ResponseVector result = forest.predict(x);

  ASSERT_EQ(y.size(), result.size());
  ASSERT_EQ(y.cols(), result.cols());
  ASSERT_EQ(y.rows(), result.rows());
  ASSERT_EQ(y, result);
}

// ---------------------------------------------------------------------------
// OOB error
// ---------------------------------------------------------------------------

TEST(OobError, PerfectSeparationGivesLowError) {
  FeatureMatrix x = DATA(Feature, 20,
      0.0f, 1.0f,
      0.1f, 2.0f,
      0.2f, 0.5f,
      0.3f, 1.5f,
      0.4f, 0.8f,
      0.5f, 1.2f,
      0.6f, 0.9f,
      0.7f, 1.8f,
      0.8f, 0.6f,
      0.9f, 1.1f,
      9.0f, 1.0f,
      9.1f, 2.0f,
      9.2f, 0.5f,
      9.3f, 1.5f,
      9.4f, 0.8f,
      9.5f, 1.2f,
      9.6f, 0.9f,
      9.7f, 1.8f,
      9.8f, 0.6f,
      9.9f, 1.1f);

  ResponseVector y = DATA(Response, 20,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

  Forest forest = Forest::train(TrainingSpecGLDA(0.0f), x, y, 50, 42);

  double err = forest.oob_error(x, y);

  ASSERT_GE(err, 0.0);
  ASSERT_LE(err, 0.1) << "Expected near-zero OOB error for perfectly separable data";
}

TEST(OobError, AllInBagReturnsNegative) {
  FeatureMatrix x = DATA(Feature, 4,
      0.0f, 0.0f,
      0.1f, 0.1f,
      9.9f, 0.0f,
      9.8f, 0.1f);

  ResponseVector y = DATA(Response, 4, 0, 0, 1, 1);

  auto condition = TreeCondition::make(
    as_projector({ 1.0f, 0.0f }),
    5.0f,
    TreeResponse::make(0),
    TreeResponse::make(1),
    nullptr,
    { 0, 1 },
    0.9f);

  Forest forest;
  forest.add_tree(
    std::make_unique<BootstrapTree>(
      std::move(condition),
      TrainingSpecGLDA::make(0.0f),
      std::vector<int>{ 0, 1, 2, 3 }));

  double err = forest.oob_error(x, y);

  ASSERT_DOUBLE_EQ(err, -1.0) << "No OOB observations, should return -1";
}

TEST(OobError, HandBuiltTreeWithKnownOob) {
  FeatureMatrix x = DATA(Feature, 6,
      0.0f, 0.5f,
      0.1f, 0.3f,
      0.2f, 0.7f,
      9.8f, 0.4f,
      9.9f, 0.6f,
      9.7f, 0.2f);

  ResponseVector y = DATA(Response, 6, 0, 0, 0, 1, 1, 1);

  auto condition = TreeCondition::make(
    as_projector({ 1.0f, 0.0f }),
    5.0f,
    TreeResponse::make(0),
    TreeResponse::make(1),
    nullptr,
    { 0, 1 },
    0.9f);

  Forest forest;
  forest.add_tree(
    std::make_unique<BootstrapTree>(
      std::move(condition),
      TrainingSpecGLDA::make(0.0f),
      std::vector<int>{ 0, 1, 4, 5 }));

  double err = forest.oob_error(x, y);

  ASSERT_DOUBLE_EQ(err, 0.0);
}

TEST(OobError, HandBuiltTreeWithOobMisclassification) {
  FeatureMatrix x = DATA(Feature, 4,
      0.0f, 0.0f,
      0.1f, 0.1f,
      9.9f, 0.0f,
      9.8f, 0.1f);

  ResponseVector y = DATA(Response, 4, 0, 1, 1, 1);

  auto condition = TreeCondition::make(
    as_projector({ 1.0f, 0.0f }),
    5.0f,
    TreeResponse::make(0),
    TreeResponse::make(1),
    nullptr,
    { 0, 1 },
    0.9f);

  Forest forest;
  forest.add_tree(
    std::make_unique<BootstrapTree>(
      std::move(condition),
      TrainingSpecGLDA::make(0.0f),
      std::vector<int>{ 0, 2 }));

  double err = forest.oob_error(x, y);

  ASSERT_DOUBLE_EQ(err, 0.5);
}

// ---------------------------------------------------------------------------
// oob_predict
// ---------------------------------------------------------------------------

TEST(OobPredict, HandBuiltTreeWithKnownOob) {
  FeatureMatrix x = DATA(Feature, 6,
      0.0f, 0.5f,
      0.1f, 0.3f,
      0.2f, 0.7f,
      9.8f, 0.4f,
      9.9f, 0.6f,
      9.7f, 0.2f);

  auto condition = TreeCondition::make(
    as_projector({ 1.0f, 0.0f }),
    5.0f,
    TreeResponse::make(0),
    TreeResponse::make(1),
    nullptr,
    { 0, 1 },
    0.9f);

  Forest forest;
  forest.add_tree(
    std::make_unique<BootstrapTree>(
      std::move(condition),
      TrainingSpecGLDA::make(0.0f),
      std::vector<int>{ 0, 1, 4, 5 }));

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
  FeatureMatrix x = DATA(Feature, 4,
      0.0f, 0.0f,
      0.1f, 0.1f,
      9.9f, 0.0f,
      9.8f, 0.1f);

  auto condition = TreeCondition::make(
    as_projector({ 1.0f, 0.0f }),
    5.0f,
    TreeResponse::make(0),
    TreeResponse::make(1),
    nullptr,
    { 0, 1 },
    0.9f);

  Forest forest;
  forest.add_tree(
    std::make_unique<BootstrapTree>(
      std::move(condition),
      TrainingSpecGLDA::make(0.0f),
      std::vector<int>{ 0, 1, 2, 3 }));

  ResponseVector preds = forest.oob_predict(x);

  ASSERT_EQ(preds.size(), 4);
  EXPECT_EQ(preds(0), -1);
  EXPECT_EQ(preds(1), -1);
  EXPECT_EQ(preds(2), -1);
  EXPECT_EQ(preds(3), -1);
}

TEST(OobPredict, HandBuiltTreeWithOobMisclassification) {
  FeatureMatrix x = DATA(Feature, 4,
      0.0f, 0.0f,
      0.1f, 0.1f,
      9.9f, 0.0f,
      9.8f, 0.1f);

  auto condition = TreeCondition::make(
    as_projector({ 1.0f, 0.0f }),
    5.0f,
    TreeResponse::make(0),
    TreeResponse::make(1),
    nullptr,
    { 0, 1 },
    0.9f);

  Forest forest;
  forest.add_tree(
    std::make_unique<BootstrapTree>(
      std::move(condition),
      TrainingSpecGLDA::make(0.0f),
      std::vector<int>{ 0, 2 }));

  ResponseVector preds = forest.oob_predict(x);

  ASSERT_EQ(preds.size(), 4);
  EXPECT_EQ(preds(0), -1) << "Row 0 in bag";
  EXPECT_EQ(preds(1), 0) << "Row 1 OOB, x[0]=0.1 < 5 -> class 0";
  EXPECT_EQ(preds(2), -1) << "Row 2 in bag";
  EXPECT_EQ(preds(3), 1) << "Row 3 OOB, x[0]=9.8 > 5 -> class 1";
}

TEST(OobPredict, ConsistentWithOobError) {
  FeatureMatrix x = DATA(Feature, 20,
      0.0f, 1.0f,
      0.1f, 2.0f,
      0.2f, 0.5f,
      0.3f, 1.5f,
      0.4f, 0.8f,
      0.5f, 1.2f,
      0.6f, 0.9f,
      0.7f, 1.8f,
      0.8f, 0.6f,
      0.9f, 1.1f,
      9.0f, 1.0f,
      9.1f, 2.0f,
      9.2f, 0.5f,
      9.3f, 1.5f,
      9.4f, 0.8f,
      9.5f, 1.2f,
      9.6f, 0.9f,
      9.7f, 1.8f,
      9.8f, 0.6f,
      9.9f, 1.1f);

  ResponseVector y = DATA(Response, 20,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

  Forest forest = Forest::train(TrainingSpecGLDA(0.0f), x, y, 50, 42);

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
