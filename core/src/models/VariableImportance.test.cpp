#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "models/VariableImportance.hpp"
#include "models/ClassificationTree.hpp"
#include "models/ClassificationForest.hpp"
#include "models/RegressionTree.hpp"
#include "models/RegressionForest.hpp"
#include "models/VIVisitor.hpp"
#include "models/Bagged.hpp"
#include "models/Forest.hpp"
#include "models/Tree.hpp"
#include "models/TreeBranch.hpp"
#include "models/TreeLeaf.hpp"
#include "models/TrainingSpec.hpp"
#include "TestSpec.hpp"
#include "models/strategies/grouping/ByCutpoint.hpp"
#include "models/strategies/leaf/MeanResponse.hpp"
#include "models/strategies/stop/MinSize.hpp"
#include "models/strategies/stop/MinVariance.hpp"
#include "models/strategies/stop/CompositeStop.hpp"
#include "models/strategies/vars/Uniform.hpp"
#include "stats/Simulation.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::pp;
using namespace ppforest2::stats;
using namespace ppforest2::types;

namespace {
  Projector as_projector(std::vector<Feature> v) {
    return Eigen::Map<Projector>(v.data(), v.size());
  }
}
// ---------------------------------------------------------------------------
// oob_indices
// ---------------------------------------------------------------------------

TEST(BaggedTreeOobIndices, ComplementOfSampleIndices) {
  // sample_indices = {0, 1, 2}, n_total = 5  =>  OOB = {3, 4}
  BaggedTree const bt(
      std::make_unique<ClassificationTree>(
          TreeLeaf::make(1), TrainingSpec::builder(types::Mode::Classification).make()
      ),
      std::vector<int>{0, 1, 2}
  );

  auto oob = bt.oob_indices(5);

  ASSERT_EQ(oob.size(), 2u);
  ASSERT_EQ(oob[0], 3);
  ASSERT_EQ(oob[1], 4);
}

TEST(BaggedTreeOobIndices, EmptyWhenAllInBag) {
  BaggedTree const bt(
      std::make_unique<ClassificationTree>(
          TreeLeaf::make(1), TrainingSpec::builder(types::Mode::Classification).make()
      ),
      std::vector<int>{0, 1, 2, 3}
  );

  auto oob = bt.oob_indices(4);
  ASSERT_TRUE(oob.empty());
}

TEST(BaggedTreeOobIndices, AllOobWhenNoneInBag) {
  BaggedTree const bt(
      std::make_unique<ClassificationTree>(
          TreeLeaf::make(1), TrainingSpec::builder(types::Mode::Classification).make()
      ),
      std::vector<int>{}
  );

  auto oob = bt.oob_indices(3);

  ASSERT_EQ(oob.size(), 3U);
  ASSERT_EQ(oob[0], 0);
  ASSERT_EQ(oob[1], 1);
  ASSERT_EQ(oob[2], 2);
}

TEST(BaggedTreeOobIndices, DuplicatesInSampleCountedOnce) {
  // With duplicates {0, 0, 1} the in-bag set is {0, 1}, OOB = {2, 3}
  BaggedTree const bt(
      std::make_unique<ClassificationTree>(
          TreeLeaf::make(1), TrainingSpec::builder(types::Mode::Classification).make()
      ),
      std::vector<int>{0, 0, 1}
  );

  auto oob = bt.oob_indices(4);

  ASSERT_EQ(oob.size(), 2U);
  ASSERT_EQ(oob[0], 2);
  ASSERT_EQ(oob[1], 3);
}

// ---------------------------------------------------------------------------
// predict_oob
// ---------------------------------------------------------------------------

TEST(BaggedTreePredictOob, MatchesRowwisePredict) {
  // Tree splits at 5.0 on x[0]: rows with x[0] < 5 -> 0, else -> 1
  BaggedTree const bt(
      std::make_unique<ClassificationTree>(
          TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F),
          TrainingSpec::builder(types::Mode::Classification).make()
      ),
      std::vector<int>{0, 1, 4, 5}
  );

  FeatureMatrix x = MAT(Feature, rows(6), 0.0F, 0.5F, 0.1F, 0.3F, 0.2F, 0.7F, 9.8F, 0.4F, 9.9F, 0.6F, 9.7F, 0.2F);

  std::vector<int> const oob_idx = {2, 3};
  OutcomeVector preds            = bt.predict_oob(x, oob_idx);

  ASSERT_EQ(preds.size(), 2);
  EXPECT_EQ(preds(0), 0) << "Row 2 has x[0]=0.2 < 5";
  EXPECT_EQ(preds(1), 1) << "Row 3 has x[0]=9.8 > 5";
}

TEST(BaggedTreePredictOob, EmptyIndicesReturnsEmptyVector) {
  BaggedTree const bt(
      std::make_unique<ClassificationTree>(
          TreeLeaf::make(1), TrainingSpec::builder(types::Mode::Classification).make()
      ),
      std::vector<int>{0, 1}
  );

  FeatureMatrix x(4, 2);
  x << 0, 0, 1, 1, 9, 9, 8, 8;

  OutcomeVector const preds = bt.predict_oob(x, std::vector<int>{});

  ASSERT_EQ(preds.size(), 0);
}

// ---------------------------------------------------------------------------
// pp_index_value stored during training
// ---------------------------------------------------------------------------

// Visitor that collects all pp_index_value fields from a tree.
struct IndexCollector : public TreeNode::Visitor {
  std::vector<Feature> values;

  void visit(TreeBranch const& node) override {
    values.push_back(node.pp_index_value);
    node.lower->accept(*this);
    node.upper->accept(*this);
  }

  void visit(TreeLeaf const& /*node*/) override {}
};

TEST(PpIndexValue, StoredAfterTraining) {
  // Two well-separated groups: col 0 discriminates.
  FeatureMatrix x = MAT(Feature, rows(6), 0.0F, 0.5F, 0.1F, 0.3F, 0.2F, 0.7F, 9.8F, 0.4F, 9.9F, 0.6F, 9.7F, 0.2F);

  OutcomeVector y = VEC(Outcome, 0, 0, 0, 1, 1, 1);

  stats::RNG rng(0, 0);
  auto tree_ptr    = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);
  Tree const& tree = *tree_ptr;

  IndexCollector collector;
  tree.root->accept(collector);

  ASSERT_FALSE(collector.values.empty());

  bool any_nonzero = false;

  for (Feature const v : collector.values) {
    if (v != Feature(0)) {
      any_nonzero = true;
      break;
    }
  }

  ASSERT_TRUE(any_nonzero) << "Expected at least one non-zero pp_index_value after training";
}

// ---------------------------------------------------------------------------
// VI2 — Tree overload (single tree, no forest wrapper)
// ---------------------------------------------------------------------------

TEST(VariableImportance2, TreeOverloadSingleNode) {
  // projector=[1,0], G_s=2, pp_index_value=0.9
  // VI2[0] = |1| / 2 = 0.5
  // VI2[1] = |0| / 2 = 0.0

  ClassificationTree const tree(
      TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F),
      test::classification_spec()
  );

  FeatureVector vi2 = tree.vi_projections(2);

  ASSERT_NEAR(vi2(0), 0.5F, 1e-5F);
  ASSERT_NEAR(vi2(1), 0.0F, 1e-5F);
}

TEST(VariableImportance2, TreeOverloadTwoNodes) {
  // Root:  projector=[1,0], G_s=3
  //   -> lower leaf
  //   -> inner: projector=[0,1], G_s=2
  //        -> leaf, leaf
  //
  // VI2[0] = |1|/3 + |0|/2 = 1/3
  // VI2[1] = |0|/3 + |1|/2 = 1/2


  ClassificationTree const tree(
      TreeBranch::make(
          as_projector({1.0F, 0.0F}),
          5.0F,
          TreeLeaf::make(0),
          TreeBranch::make(as_projector({0.0F, 1.0F}), 3.0F, TreeLeaf::make(1), TreeLeaf::make(2), {1, 2}, 0.6F),
          {0, 1, 2},
          0.8F
      ),
      test::classification_spec()
  );

  FeatureVector vi2 = tree.vi_projections(2);

  ASSERT_NEAR(vi2(0), 1.0F / 3.0F, 1e-5F);
  ASSERT_NEAR(vi2(1), 1.0F / 2.0F, 1e-5F);
}

TEST(VariableImportance2, TreeOverloadWithScale) {
  // projector=[0.5, 0.2], G_s=2
  // scale=[2.0, 3.0]
  // VI2[0] = |0.5| * 2.0 / 2 = 0.5
  // VI2[1] = |0.2| * 3.0 / 2 = 0.3


  ClassificationTree const tree(
      TreeBranch::make(as_projector({0.5F, 0.2F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F),
      test::classification_spec()
  );

  FeatureVector scale(2);
  scale << 2.0F, 3.0F;

  FeatureVector vi2 = tree.vi_projections(2, &scale);

  ASSERT_NEAR(vi2(0), 0.5F, 1e-5F);
  ASSERT_NEAR(vi2(1), 0.3F, 1e-5F);
}

// ---------------------------------------------------------------------------
// VI2 — hand-built single-node tree (forest wrapper)
// ---------------------------------------------------------------------------

TEST(VariableImportance2, HandBuiltSingleNodeTree) {
  // Build a forest with a single, manually constructed bootstrap tree.
  //   projector = [1, 0], pp_index_value = 0.9, G_s = 2
  // Expected VI2[j] = |a_j| / G_s:
  //   VI2[0] = 1.0 / 2 = 0.5
  //   VI2[1] = 0.0 / 2 = 0.0
  ClassificationForest forest;
  forest.add_tree(std::make_unique<BaggedTree>(
      std::make_unique<ClassificationTree>(
          TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F),
          TrainingSpec::builder(types::Mode::Classification).make()
      ),
      std::vector<int>{0, 1, 2, 3}
  ));

  FeatureVector vi2 = forest.vi_projections(2);

  ASSERT_NEAR(vi2(0), 0.5F, 1e-5F);
  ASSERT_NEAR(vi2(1), 0.0F, 1e-5F);
}

TEST(VariableImportance2, HandBuiltTwoNodeTree) {
  // Root:  projector=[1,0], G_s=3, pp_index_value=0.8
  //   -> lower leaf
  //   -> inner: projector=[0,1], G_s=2, pp_index_value=0.6
  //        -> leaf, leaf
  //
  // VI2[0] = (1/B) * (|1|/3 + |0|/2) = 1/3
  // VI2[1] = (1/B) * (|0|/3 + |1|/2) = 1/2
  ClassificationForest forest;
  forest.add_tree(std::make_unique<BaggedTree>(
      std::make_unique<ClassificationTree>(
          TreeBranch::make(
              as_projector({1.0F, 0.0F}),
              5.0F,
              TreeLeaf::make(0),
              TreeBranch::make(as_projector({0.0F, 1.0F}), 3.0F, TreeLeaf::make(1), TreeLeaf::make(2), {1, 2}, 0.6F),
              {0, 1, 2},
              0.8F
          ),
          TrainingSpec::builder(types::Mode::Classification).make()
      ),
      std::vector<int>{0, 1, 2, 3}
  ));

  FeatureVector vi2 = forest.vi_projections(2);

  ASSERT_NEAR(vi2(0), 1.0F / 3.0F, 1e-5F);
  ASSERT_NEAR(vi2(1), 1.0F / 2.0F, 1e-5F);
}

// ---------------------------------------------------------------------------
// VI3 — hand-built single-node tree
// ---------------------------------------------------------------------------

TEST(VariableImportance3, HandBuiltSingleNodeTree) {
  // projector=[1,0], pp_index_value=0.9, G_s=2. All 4 rows are in-bag, so
  // the single tree has empty OOB.
  //
  // A tree that saw every training row during fitting has no out-of-sample
  // view to evaluate against, so it can't participate in the OOB-weighted
  // VI3 average and is dropped entirely. With only one tree in this forest
  // that leaves `valid_trees = 0`, the divisor is zero, and VI3 stays at
  // its zero initialization.

  FeatureMatrix x = MAT(Feature, rows(4), 0.0F, 0.0F, 0.1F, 0.1F, 9.9F, 0.0F, 9.8F, 0.1F);
  OutcomeVector y = VEC(Outcome, 0, 0, 1, 1);

  ClassificationForest forest;
  forest.add_tree(std::make_unique<BaggedTree>(
      std::make_unique<ClassificationTree>(
          TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F),
          TrainingSpec::builder(types::Mode::Classification).make()
      ),
      std::vector<int>{0, 1, 2, 3}
  )); // all in-bag

  FeatureVector vi3 = forest.vi_weighted_projections(x, y.cast<Outcome>());

  ASSERT_NEAR(vi3(0), 0.0F, 1e-5F);
  ASSERT_NEAR(vi3(1), 0.0F, 1e-5F);
}

// ---------------------------------------------------------------------------
// VI1 — discriminating variable gets highest importance
// ---------------------------------------------------------------------------

TEST(VariableImportance1, DiscriminatingVariableHighestImportance) {
  // Col 0 perfectly separates the two groups.
  // Col 1 is pure noise (same values in both groups).
  FeatureMatrix x =
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

  auto forest_ptr      = Forest::train(TrainingSpec::builder(types::Mode::Classification).size(10).build(), x, y);
  Forest const& forest = *forest_ptr;

  FeatureVector vi1 = forest.vi_permuted(x, y.cast<Outcome>(), 0);

  ASSERT_GT(vi1(0), vi1(1)) << "Expected discriminating variable (col 0) to have higher VI1 than noise (col 1). "
                            << "VI1[0]=" << vi1(0) << ", VI1[1]=" << vi1(1);
}

// ---------------------------------------------------------------------------
// Regression VI — informative features should outrank noise
// ---------------------------------------------------------------------------

namespace {
  TrainingSpec make_reg_spec(int size, int n_vars_val, int seed) {
    return TrainingSpec::builder(types::Mode::Regression)
        .size(size)
        .seed(seed)
        .threads(1)
        .pp(pp::pda(Feature(0)))
        .vars(vars::uniform(n_vars_val))
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .stop(stop::any({stop::min_size(5), stop::min_variance(0.01F)}))
        .build();
  }
}

TEST(VariableImportance1Regression, InformativeFeaturesOutrankNoise) {
  // simulate_regression generates y = sum(j+1 * x_j) for the first
  // n_informative columns plus noise. With n_informative=2 and p=5,
  // columns 0 and 1 drive y; columns 2..4 are pure noise.
  RNG rng(42);
  RegressionSimulationParams params;
  params.n_informative = 2;
  params.noise_sd      = 0.1F;
  params.feature_sd    = 1.0F;
  auto data            = simulate_regression(200, 5, rng, params);

  auto spec   = make_reg_spec(20, 3, 0);
  auto forest = Forest::train(spec, data.x, data.y);

  FeatureVector vi1 = forest->vi_permuted(data.x, data.y, 0);

  // Both informative variables should outrank every noise variable.
  for (int j = 2; j < 5; ++j) {
    EXPECT_GT(vi1(0), vi1(j)) << "VI1[0]=" << vi1(0) << " not > VI1[" << j << "]=" << vi1(j);
    EXPECT_GT(vi1(1), vi1(j)) << "VI1[1]=" << vi1(1) << " not > VI1[" << j << "]=" << vi1(j);
  }
}

TEST(VariableImportance3Regression, InformativeFeaturesDominateAsAGroup) {
  RNG rng(42);
  RegressionSimulationParams params;
  params.n_informative = 2;
  params.noise_sd      = 0.1F;
  params.feature_sd    = 1.0F;
  auto data            = simulate_regression(200, 5, rng, params);

  auto spec   = make_reg_spec(20, 3, 0);
  auto forest = Forest::train(spec, data.x, data.y);

  FeatureVector scale = stats::sd(data.x);
  scale               = (scale.array() > Feature(0)).select(scale, Feature(1));

  FeatureVector vi3 = forest->vi_weighted_projections(data.x, data.y, &scale);

  // VI3 is structural (projection coefficient × tree-quality weight); individual
  // noise features can occasionally outrank a single informative feature by chance.
  // The aggregate signal should still favour the informative set.
  Feature const total_informative = vi3(0) + vi3(1);
  Feature const total_noise       = vi3(2) + vi3(3) + vi3(4);

  EXPECT_GT(total_informative, total_noise)
      << "Sum(VI3[0..1])=" << total_informative << " not > Sum(VI3[2..4])=" << total_noise;
}

TEST(VariableImportanceRegression, BundleConvenienceFillsAllFields) {
  RNG rng(42);
  auto data = simulate_regression(100, 4, rng);

  auto spec   = make_reg_spec(5, 2, 0);
  auto forest = Forest::train(spec, data.x, data.y);

  VariableImportance vi = forest->variable_importance(data.x, data.y, 0);

  int const p = static_cast<int>(data.x.cols());
  EXPECT_EQ(vi.permuted.size(), p);
  EXPECT_EQ(vi.projections.size(), p);
  EXPECT_EQ(vi.weighted_projections.size(), p);
  EXPECT_EQ(vi.scale.size(), p);
}
