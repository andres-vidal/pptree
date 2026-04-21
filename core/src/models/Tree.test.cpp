#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "models/Tree.hpp"
#include "models/ClassificationTree.hpp"
#include "models/TreeBranch.hpp"
#include "models/TreeLeaf.hpp"

#include "models/TrainingSpec.hpp"
#include "TestSpec.hpp"

#include "stats/Simulation.hpp"
#include "stats/Stats.hpp"
#include "utils/Macros.hpp"

#include "serialization/Json.hpp"

using namespace ppforest2;
using namespace ppforest2::pp;
using namespace ppforest2::stats;
using namespace ppforest2::types;
using namespace ppforest2::math;
using namespace ppforest2::serialization;
namespace {
  Projector as_projector(std::vector<Feature> v) {
    return Eigen::Map<Projector>(v.data(), v.size());
  }
}

TEST(TreeLeaf, EqualsEqualResponses) {
  TreeLeaf const r1(1);
  TreeLeaf const r2(1);

  ASSERT_TRUE(r1 == r2);
}

TEST(TreeLeaf, EqualsDifferentResponses) {
  TreeLeaf const r1(1);
  TreeLeaf const r2(2);

  ASSERT_FALSE(r1 == r2);
}

TEST(TreeBranch, EqualsEqualConditions) {
  TreeBranch const c1(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));
  TreeBranch const c2(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(TreeBranch, EqualsCollinearProjectors) {
  TreeBranch const c1(as_projector({1.0, 1.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));
  TreeBranch const c2(as_projector({2.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(TreeBranch, EqualsApproximateThresholds) {
  TreeBranch const c1(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));
  TreeBranch const c2(as_projector({1.0, 2.0}), 3.000000000000001, TreeLeaf::make(1), TreeLeaf::make(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(TreeBranch, EqualsNonCollinearProjectors) {
  TreeBranch const c1(as_projector({1.0, 0.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));
  TreeBranch const c2(as_projector({0.0, 1.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeBranch, EqualsDifferentThresholds) {
  TreeBranch const c1(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));
  TreeBranch const c2(as_projector({1.0, 2.0}), 4.0, TreeLeaf::make(1), TreeLeaf::make(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeBranch, EqualsDifferentResponses) {
  TreeBranch const c1(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));
  TreeBranch const c2(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(3));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeBranch, EqualsDifferentStructures) {
  TreeBranch const c1(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2));
  TreeBranch const c2(
      as_projector({1.0, 2.0}),
      3.0,
      TreeLeaf::make(1),
      TreeBranch::make(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2))
  );

  ASSERT_FALSE(c1 == c2);
}

TEST(Tree, EqualsEqualTrees) {
  ClassificationTree const t1(
      TreeBranch::make(
          as_projector({1.0, 2.0}),
          3.0,
          TreeLeaf::make(1),
          TreeBranch::make(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2))
      ),
      test::classification_spec()
  );

  ClassificationTree const t2(
      TreeBranch::make(
          as_projector({1.0, 2.0}),
          3.0,
          TreeLeaf::make(1),
          TreeBranch::make(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2))
      ),
      test::classification_spec()
  );

  ASSERT_TRUE(t1 == t2);
}

TEST(Tree, EqualsDifferentTrees) {
  ClassificationTree const t1(TreeBranch::make(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2)), test::classification_spec());

  ClassificationTree const t2(TreeBranch::make(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(3)), test::classification_spec());

  ASSERT_FALSE(t1 == t2);
}

TEST(Tree, TrainLDAUnivariateTwoGroups) {
  FeatureMatrix x = MAT(Feature, rows(10), 1, 1, 1, 1, 1, 2, 2, 2, 2, 2);

  OutcomeVector const y = VEC(Outcome, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1);

  stats::RNG rng(0);

  auto const result_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);
  Tree const& result = *result_ptr;

  OutcomeVector const predictions = result.predict(x);

  ASSERT_EQ(error_rate(predictions, y), 0.0) << "Tree should achieve 0% training error on well-separated 2-group data";
}

TEST(Tree, TrainLDAUnivariateThreeGroups) {
  FeatureMatrix x = MAT(Feature, rows(15), 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3);

  OutcomeVector const y = VEC(Outcome, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2);

  stats::RNG rng(0);

  auto const result_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);
  Tree const& result = *result_ptr;

  OutcomeVector const predictions = result.predict(x);

  ASSERT_EQ(error_rate(predictions, y), 0.0) << "Tree should achieve 0% training error on well-separated 3-group data";
}

TEST(Tree, TrainLDAMultivariateTwoGroups) {
  FeatureMatrix x =
      MAT(Feature,
          rows(10),
          1,
          0,
          1,
          1,
          1,
          1,
          0,
          0,
          1,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          4,
          0,
          0,
          1,
          4,
          0,
          0,
          2,
          4,
          0,
          0,
          3,
          4,
          1,
          0,
          1,
          4,
          0,
          1,
          1,
          4,
          0,
          1,
          2);

  OutcomeVector const y = VEC(Outcome, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

  stats::RNG rng(0);

  auto const result_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);
  Tree const& result = *result_ptr;

  OutcomeVector const predictions = result.predict(x);

  ASSERT_EQ(error_rate(predictions, y), 0.0) << "Tree should achieve 0% training error on well-separated 2-group data";
}

TEST(Tree, TrainLDAMultivariateThreeGroupsProperties) {
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

  OutcomeVector y =
      VEC(Outcome, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);

  stats::RNG rng(0);

  auto const result_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);
  Tree const& result = *result_ptr;

  // Property: tree should predict training data perfectly (well-separated groups)
  OutcomeVector const predictions = result.predict(x);

  ASSERT_EQ(error_rate(predictions, y), 0.0) << "Tree should achieve 0% training error on well-separated 3-group data";
}

TEST(Tree, TrainPDAMultivariateTwoGroupsProperties) {
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

  OutcomeVector const y = VEC(Outcome, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

  stats::RNG rng(0);

  auto const result_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).pp(pp::pda(0.5F)).build(), x, y, rng);
  Tree const& result = *result_ptr;

  // Property: tree should predict training data perfectly
  OutcomeVector const predictions = result.predict(x);

  ASSERT_EQ(error_rate(predictions, y), 0.0)
      << "PDA tree should achieve 0% training error on well-separated 2-group data";
}

TEST(TreeSimulation, PerfectSeparationZeroError) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 200.0F;
  params.sd              = 1.0F;

  auto data = simulate(90, 4, 3, rng, params);

  RNG train_rng(0);
  auto const tree_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), data.x, data.y, train_rng);
  Tree const& tree = *tree_ptr;

  OutcomeVector const predictions = tree.predict(data.x);
  ASSERT_EQ(error_rate(predictions, data.y), 0.0) << "Tree should achieve 0% error on perfectly separated data";
}

TEST(TreeSimulation, HighOverlapBoundedError) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 5.0F;
  params.sd              = 50.0F;

  auto data = simulate(200, 4, 3, rng, params);

  RNG train_rng(0);
  auto const tree_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), data.x, data.y, train_rng);
  Tree const& tree = *tree_ptr;

  OutcomeVector const predictions = tree.predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.80) << "Tree error rate on highly overlapping data should be bounded";
}

TEST(TreeSimulation, ManyClasses) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 50.0F;
  params.sd              = 10.0F;

  auto data = simulate(300, 4, 10, rng, params);

  RNG train_rng(0);
  auto const tree_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), data.x, data.y, train_rng);
  Tree const& tree = *tree_ptr;

  OutcomeVector const predictions = tree.predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.50) << "Tree should handle 10 groups with reasonable error";
}

TEST(TreeSimulation, HighDimensionality) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 50.0F;
  params.sd              = 10.0F;

  auto data = simulate(100, 50, 3, rng, params);

  RNG train_rng(0);
  auto const tree_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), data.x, data.y, train_rng);
  Tree const& tree = *tree_ptr;

  OutcomeVector const predictions = tree.predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.20) << "Tree should handle high-dimensional data (p=50)";
}

TEST(TreeSimulation, SingleFeature) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 50.0F;
  params.sd              = 10.0F;

  auto data = simulate(100, 1, 2, rng, params);

  RNG train_rng(0);
  auto const tree_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), data.x, data.y, train_rng);
  Tree const& tree = *tree_ptr;

  OutcomeVector const predictions = tree.predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.10) << "Tree should handle univariate data";
}

TEST(TreeSimulation, PDAOnOverlappingData) {
  RNG rng(0);
  SimulationParams params;
  params.mean_separation = 10.0F;
  params.sd              = 20.0F;

  auto data = simulate(200, 4, 3, rng, params);

  RNG train_rng(0);
  auto const tree_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).pp(pp::pda(0.5F)).build(), data.x, data.y, train_rng);
  Tree const& tree = *tree_ptr;

  OutcomeVector const predictions = tree.predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.80) << "PDA tree should produce bounded error on noisy data";
}

TEST(TreeSimulation, Deterministic) {
  RNG rng(0);
  auto data = simulate(90, 4, 3, rng);

  RNG rng1(0);
  auto const t1_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), data.x, data.y, rng1);
  Tree const& t1 = *t1_ptr;

  RNG rng2(0);
  auto const t2_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), data.x, data.y, rng2);
  Tree const& t2 = *t2_ptr;

  ASSERT_EQ(t1, t2) << "Same seed should produce identical trees";
}

TEST(Tree, PredictDataColumnUnivariateTwoGroups) {
  ClassificationTree const tree(TreeBranch::make(as_projector({1.0}), 1.5, TreeLeaf::make(0), TreeLeaf::make(1)), test::classification_spec());


  FeatureVector input = VEC(Feature, 1.0);
  ASSERT_EQ(tree.predict(input), 0);

  input = VEC(Feature, 2.0);
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(Tree, PredictDataColumnUnivariateThreeGroups) {
  ClassificationTree const tree(
      TreeBranch::make(
          as_projector({1.0}),
          1.75,
          TreeLeaf::make(0),
          TreeBranch::make(as_projector({1.0}), 2.5, TreeLeaf::make(1), TreeLeaf::make(2))
      ),
      test::classification_spec()
  );

  FeatureVector input = VEC(Feature, 1.0);
  ASSERT_EQ(tree.predict(input), 0);

  input = VEC(Feature, 2.0);
  ASSERT_EQ(tree.predict(input), 1);

  input = VEC(Feature, 3.0);
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(Tree, PredictDataColumnMultivariateTwoGroups) {
  ClassificationTree const tree =
      ClassificationTree(TreeBranch::make(as_projector({1.0, 0.0, 0.0, 0.0}), 2.5, TreeLeaf::make(0), TreeLeaf::make(1)), test::classification_spec());

  FeatureVector input = VEC(Feature, 1, 0, 1, 1);
  ASSERT_EQ(tree.predict(input), 0);

  input = VEC(Feature, 4, 0, 0, 1);
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(Tree, PredictDataColumnMultivariateThreeGroups) {
  ClassificationTree const tree(
      TreeBranch::make(
          as_projector({0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0}),
          4.118438837901864,
          TreeBranch::make(as_projector({0.0, 1.0, 0.0, 0.0, 0.0}), 2.5, TreeLeaf::make(0), TreeLeaf::make(1)),
          TreeLeaf::make(2)
      ),
      test::classification_spec()
  );

  FeatureVector input = VEC(Feature, 1, 0, 0, 1, 1);
  ASSERT_EQ(tree.predict(input), 0);

  input = VEC(Feature, 2, 5, 0, 0, 1);
  ASSERT_EQ(tree.predict(input), 1);

  input = VEC(Feature, 9, 8, 0, 0, 1);
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(Tree, PredictDataUnivariateTwoGroups) {
  ClassificationTree const tree(TreeBranch::make(as_projector({1.0}), 1.5, TreeLeaf::make(0), TreeLeaf::make(1)), test::classification_spec());

  FeatureMatrix const input = MAT(Feature, rows(2), 1.0, 2.0);

  OutcomeVector const result   = tree.predict(input);
  OutcomeVector const expected = VEC(Outcome, 0, 1);

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataUnivariateThreeGroups) {
  ClassificationTree const tree(
      TreeBranch::make(
          as_projector({1.0}),
          1.75,
          TreeLeaf::make(0),
          TreeBranch::make(as_projector({1.0}), 2.5, TreeLeaf::make(1), TreeLeaf::make(2))
      ),
      test::classification_spec()
  );

  FeatureMatrix const input = MAT(Feature, rows(3), 1.0, 2.0, 3.0);

  OutcomeVector const result   = tree.predict(input);
  OutcomeVector const expected = VEC(Outcome, 0, 1, 2);

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataMultivariateTwoGroups) {
  ClassificationTree const tree =
      ClassificationTree(TreeBranch::make(as_projector({1.0, 0.0, 0.0, 0.0}), 2.5, TreeLeaf::make(0), TreeLeaf::make(1)), test::classification_spec());

  FeatureMatrix const input = MAT(Feature, rows(2), 1, 0, 1, 1, 4, 0, 0, 1);

  OutcomeVector const result   = tree.predict(input);
  OutcomeVector const expected = VEC(Outcome, 0, 1);

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataMultivariateThreeGroups) {
  ClassificationTree tree(
      TreeBranch::make(
          as_projector({0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0}),
          4.118438837901864,
          TreeBranch::make(as_projector({0.0, 1.0, 0.0, 0.0, 0.0}), 2.5, TreeLeaf::make(0), TreeLeaf::make(1)),
          TreeLeaf::make(2)
      ),
      test::classification_spec()
  );

  FeatureMatrix const input = MAT(Feature, rows(3), 1, 0, 0, 1, 1, 2, 5, 0, 0, 1, 9, 8, 0, 0, 1);

  OutcomeVector const result   = tree.predict(input);
  OutcomeVector const expected = VEC(Outcome, 0, 1, 2);

  ASSERT_EQ(result, expected) << "Tree should predict the correct groups for the input data";
}

TEST(Tree, PredictProportionsTwoGroups) {
  ClassificationTree const tree =
      ClassificationTree(TreeBranch::make(as_projector({1.0}), 1.5, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}), test::classification_spec());

  FeatureMatrix const input = MAT(Feature, rows(2), 1.0, 2.0);

  FeatureMatrix const result = tree.predict(input, Proportions{});

  ASSERT_EQ(result.rows(), 2);
  ASSERT_EQ(result.cols(), 2);

  // Row 0 predicts group 0 → [1, 0]
  EXPECT_FLOAT_EQ(result(0, 0), 1.0);
  EXPECT_FLOAT_EQ(result(0, 1), 0.0);

  // Row 1 predicts group 1 → [0, 1]
  EXPECT_FLOAT_EQ(result(1, 0), 0.0);
  EXPECT_FLOAT_EQ(result(1, 1), 1.0);
}

TEST(Tree, PredictProportionsThreeGroups) {
  ClassificationTree const tree(
      TreeBranch::make(
          as_projector({1.0}),
          1.75,
          TreeLeaf::make(0),
          TreeBranch::make(as_projector({1.0}), 2.5, TreeLeaf::make(1), TreeLeaf::make(2), {1, 2}),
          {0, 1, 2}
      ),
      test::classification_spec()
  );

  FeatureMatrix const input = MAT(Feature, rows(3), 1.0, 2.0, 3.0);

  FeatureMatrix const result = tree.predict(input, Proportions{});

  ASSERT_EQ(result.rows(), 3);
  ASSERT_EQ(result.cols(), 3);

  // Each row is one-hot for the predicted group
  EXPECT_FLOAT_EQ(result(0, 0), 1.0);
  EXPECT_FLOAT_EQ(result(0, 1), 0.0);
  EXPECT_FLOAT_EQ(result(0, 2), 0.0);

  EXPECT_FLOAT_EQ(result(1, 0), 0.0);
  EXPECT_FLOAT_EQ(result(1, 1), 1.0);
  EXPECT_FLOAT_EQ(result(1, 2), 0.0);

  EXPECT_FLOAT_EQ(result(2, 0), 0.0);
  EXPECT_FLOAT_EQ(result(2, 1), 0.0);
  EXPECT_FLOAT_EQ(result(2, 2), 1.0);
}

TEST(Tree, PredictProportionsRowsSumToOne) {
  ClassificationTree const tree(
      TreeBranch::make(as_projector({1.0, 0.0, 0.0, 0.0}), 2.5, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}), test::classification_spec()
  );

  FeatureMatrix const input = MAT(Feature, rows(4), 1, 0, 1, 1, 4, 0, 0, 1, 2, 1, 0, 0, 3, 0, 1, 0);

  FeatureMatrix const result = tree.predict(input, Proportions{});

  ASSERT_EQ(result.rows(), 4);
  ASSERT_EQ(result.cols(), 2);

  for (int i = 0; i < result.rows(); ++i) {
    EXPECT_FLOAT_EQ(result.row(i).sum(), Feature(1)) << "Row " << i << " should sum to 1.0";
  }
}

TEST(Tree, PredictProportionsMatchesPredictions) {
  ClassificationTree const tree(
      TreeBranch::make(
          as_projector({0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0}),
          4.118438837901864,
          TreeBranch::make(as_projector({0.0, 1.0, 0.0, 0.0, 0.0}), 2.5, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}),
          TreeLeaf::make(2),
          {0, 1, 2}
      ),
      test::classification_spec()
  );

  FeatureMatrix const input = MAT(Feature, rows(3), 1, 0, 0, 1, 1, 2, 5, 0, 0, 1, 9, 8, 0, 0, 1);

  OutcomeVector predictions = tree.predict(input);
  FeatureMatrix proportions = tree.predict(input, Proportions{});

  ASSERT_EQ(proportions.rows(), 3);
  ASSERT_EQ(proportions.cols(), 3);

  // The column with 1.0 in each row must match the predicted group
  for (int i = 0; i < input.rows(); ++i) {
    EXPECT_FLOAT_EQ(proportions(i, static_cast<int>(predictions(i))), Feature(1))
        << "Row " << i << ": proportions should have 1.0 in column for predicted group";
  }
}

// ---------------------------------------------------------------------------
// Edge cases — "doesn't blow up" tests
// ---------------------------------------------------------------------------

TEST(TreeEdgeCase, ConstantFeatureColumn) {
  FeatureMatrix x = MAT(Feature, rows(6), 5, 1, 5, 2, 5, 3, 5, 7, 5, 8, 5, 9);
  OutcomeVector const y = VEC(Outcome, 0, 0, 0, 1, 1, 1);

  stats::RNG rng(0);
  auto const tree_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);
  Tree const& tree = *tree_ptr;

  ASSERT_EQ_DATA(tree.predict(x), VEC(Outcome, 0, 0, 0, 0, 0, 0));
}

TEST(TreeEdgeCase, SingleObservationPerGroup) {
  FeatureMatrix x = MAT(Feature, rows(2), 1, 0, 0, 1);
  OutcomeVector const y = VEC(Outcome, 0, 1);

  stats::RNG rng(0);
  auto const tree_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);
  Tree const& tree = *tree_ptr;

  ASSERT_EQ_DATA(tree.predict(x), VEC(Outcome, 0, 1));
}

TEST(TreeEdgeCase, MinimalDataset) {
  FeatureMatrix x = MAT(Feature, rows(2), 1, 9);
  OutcomeVector const y = VEC(Outcome, 0, 1);

  stats::RNG rng(0);
  auto const tree_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);
  Tree const& tree = *tree_ptr;

  ASSERT_EQ_DATA(tree.predict(x), VEC(Outcome, 0, 1));
}

TEST(TreeEdgeCase, ExtremeImbalance) {
  // clang-format off
  FeatureMatrix x = MAT(Feature, rows(20),
    0, 0,  1, 1,  2, 2,  3, 0,  4, 1,
    0, 2,  1, 0,  2, 1,  3, 2,  4, 0,
    0, 1,  1, 2,  2, 0,  3, 1,  4, 2,
    0, 0,  1, 1,  2, 2,  90, 90,  91, 91);
  OutcomeVector const y = VEC(Outcome,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1);
  // clang-format on

  stats::RNG rng(0);
  auto const tree_ptr = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);
  Tree const& tree = *tree_ptr;

  ASSERT_EQ_DATA(tree.predict(x), y.cast<Outcome>());
}
