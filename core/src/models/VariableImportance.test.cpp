#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "models/VariableImportance.hpp"
#include "models/VIVisitor.hpp"
#include "models/BootstrapTree.hpp"
#include "models/Forest.hpp"
#include "models/Tree.hpp"
#include "models/TreeBranch.hpp"
#include "models/TreeLeaf.hpp"
#include "models/TrainingSpec.hpp"
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

TEST(BootstrapTreeOobIndices, ComplementOfSampleIndices) {
  // sample_indices = {0, 1, 2}, n_total = 5  =>  OOB = {3, 4}
  BootstrapTree const bt(TreeLeaf::make(1), TrainingSpec::builder().make(), std::vector<int>{0, 1, 2});

  auto oob = bt.oob_indices(5);

  ASSERT_EQ(oob.size(), 2u);
  ASSERT_EQ(oob[0], 3);
  ASSERT_EQ(oob[1], 4);
}

TEST(BootstrapTreeOobIndices, EmptyWhenAllInBag) {
  BootstrapTree const bt(TreeLeaf::make(1), TrainingSpec::builder().make(), std::vector<int>{0, 1, 2, 3});

  auto oob = bt.oob_indices(4);
  ASSERT_TRUE(oob.empty());
}

TEST(BootstrapTreeOobIndices, AllOobWhenNoneInBag) {
  BootstrapTree const bt(TreeLeaf::make(1), TrainingSpec::builder().make(), std::vector<int>{});

  auto oob = bt.oob_indices(3);

  ASSERT_EQ(oob.size(), 3U);
  ASSERT_EQ(oob[0], 0);
  ASSERT_EQ(oob[1], 1);
  ASSERT_EQ(oob[2], 2);
}

TEST(BootstrapTreeOobIndices, DuplicatesInSampleCountedOnce) {
  // With duplicates {0, 0, 1} the in-bag set is {0, 1}, OOB = {2, 3}
  BootstrapTree const bt(TreeLeaf::make(1), TrainingSpec::builder().make(), std::vector<int>{0, 0, 1});

  auto oob = bt.oob_indices(4);

  ASSERT_EQ(oob.size(), 2U);
  ASSERT_EQ(oob[0], 2);
  ASSERT_EQ(oob[1], 3);
}

// ---------------------------------------------------------------------------
// predict_oob
// ---------------------------------------------------------------------------

TEST(BootstrapTreePredictOob, MatchesRowwisePredict) {
  // Tree splits at 5.0 on x[0]: rows with x[0] < 5 -> 0, else -> 1
  BootstrapTree const bt(
      TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F),
      TrainingSpec::builder().make(),
      std::vector<int>{0, 1, 4, 5}
  );

  FeatureMatrix const x = MAT(Feature, rows(6), 0.0F, 0.5F, 0.1F, 0.3F, 0.2F, 0.7F, 9.8F, 0.4F, 9.9F, 0.6F, 9.7F, 0.2F);

  std::vector<int> const oob_idx = {2, 3};
  OutcomeVector preds            = bt.predict_oob(x, oob_idx);

  ASSERT_EQ(preds.size(), 2);
  EXPECT_EQ(preds(0), 0) << "Row 2 has x[0]=0.2 < 5";
  EXPECT_EQ(preds(1), 1) << "Row 3 has x[0]=9.8 > 5";
}

TEST(BootstrapTreePredictOob, EmptyIndicesReturnsEmptyVector) {
  BootstrapTree const bt(TreeLeaf::make(1), TrainingSpec::builder().make(), std::vector<int>{0, 1});

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
  FeatureMatrix const x = MAT(Feature, rows(6), 0.0F, 0.5F, 0.1F, 0.3F, 0.2F, 0.7F, 9.8F, 0.4F, 9.9F, 0.6F, 9.7F, 0.2F);

  OutcomeVector const y = VEC(Outcome, 0, 0, 0, 1, 1, 1);

  stats::RNG rng(0, 0);
  Tree const tree = Tree::train(TrainingSpec::builder().build(), x, y, rng);

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

  Tree const tree(
      TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F), nullptr
  );

  FeatureVector vi2 = variable_importance_projections(tree, 2);

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


  Tree const tree(
      TreeBranch::make(
          as_projector({1.0F, 0.0F}),
          5.0F,
          TreeLeaf::make(0),
          TreeBranch::make(as_projector({0.0F, 1.0F}), 3.0F, TreeLeaf::make(1), TreeLeaf::make(2), {1, 2}, 0.6F),
          {0, 1, 2},
          0.8F
      ),
      nullptr
  );

  FeatureVector vi2 = variable_importance_projections(tree, 2);

  ASSERT_NEAR(vi2(0), 1.0F / 3.0F, 1e-5F);
  ASSERT_NEAR(vi2(1), 1.0F / 2.0F, 1e-5F);
}

TEST(VariableImportance2, TreeOverloadWithScale) {
  // projector=[0.5, 0.2], G_s=2
  // scale=[2.0, 3.0]
  // VI2[0] = |0.5| * 2.0 / 2 = 0.5
  // VI2[1] = |0.2| * 3.0 / 2 = 0.3


  Tree const tree(
      TreeBranch::make(as_projector({0.5F, 0.2F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F), nullptr
  );

  FeatureVector scale(2);
  scale << 2.0F, 3.0F;

  FeatureVector vi2 = variable_importance_projections(tree, 2, &scale);

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
  Forest forest;
  forest.add_tree(std::make_unique<BootstrapTree>(
      TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F),
      TrainingSpec::builder().make(),
      std::vector<int>{0, 1, 2, 3}
  ));

  FeatureVector vi2 = variable_importance_projections(forest, 2);

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
  Forest forest;
  forest.add_tree(std::make_unique<BootstrapTree>(
      TreeBranch::make(
          as_projector({1.0F, 0.0F}),
          5.0F,
          TreeLeaf::make(0),
          TreeBranch::make(as_projector({0.0F, 1.0F}), 3.0F, TreeLeaf::make(1), TreeLeaf::make(2), {1, 2}, 0.6F),
          {0, 1, 2},
          0.8F
      ),
      TrainingSpec::builder().make(),
      std::vector<int>{0, 1, 2, 3}
  ));

  FeatureVector vi2 = variable_importance_projections(forest, 2);

  ASSERT_NEAR(vi2(0), 1.0F / 3.0F, 1e-5F);
  ASSERT_NEAR(vi2(1), 1.0F / 2.0F, 1e-5F);
}

// ---------------------------------------------------------------------------
// VI3 — hand-built single-node tree
// ---------------------------------------------------------------------------

TEST(VariableImportance3, HandBuiltSingleNodeTree) {
  // projector=[1,0], pp_index_value=0.9, G_s=2
  // All rows are in-bag => OOB empty => e_k = 0 => weight = 1
  //
  // vi3_contributions[0] = I_s * |a_0| = 0.9 * 1.0 = 0.9
  // vi3_contributions[1] = 0.9 * 0.0 = 0.0
  //
  // denom = B * (G-1) = 1 * (2-1) = 1
  // VI3[0] = 1.0 * 0.9 / 1 = 0.9,  VI3[1] = 0.0

  FeatureMatrix const x = MAT(Feature, rows(4), 0.0F, 0.0F, 0.1F, 0.1F, 9.9F, 0.0F, 9.8F, 0.1F);
  OutcomeVector const y = VEC(Outcome, 0, 0, 1, 1);

  Forest forest;
  forest.add_tree(std::make_unique<BootstrapTree>(
      TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F),
      TrainingSpec::builder().make(),
      std::vector<int>{0, 1, 2, 3}
  )); // all in-bag

  FeatureVector vi3 = variable_importance_weighted_projections(forest, x, y);

  ASSERT_NEAR(vi3(0), 0.9F, 1e-5F);
  ASSERT_NEAR(vi3(1), 0.0F, 1e-5F);
}

// ---------------------------------------------------------------------------
// VI1 — discriminating variable gets highest importance
// ---------------------------------------------------------------------------

TEST(VariableImportance1, DiscriminatingVariableHighestImportance) {
  // Col 0 perfectly separates the two groups.
  // Col 1 is pure noise (same values in both groups).
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

  Forest const forest = Forest::train(TrainingSpec::builder().size(10).build(), x, y);

  FeatureVector vi1 = variable_importance_permuted(forest, x, y, 0);

  ASSERT_GT(vi1(0), vi1(1)) << "Expected discriminating variable (col 0) to have higher VI1 than noise (col 1). "
                            << "VI1[0]=" << vi1(0) << ", VI1[1]=" << vi1(1);
}
