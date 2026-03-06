#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "models/VariableImportance.hpp"
#include "models/VIVisitor.hpp"
#include "models/BootstrapTree.hpp"
#include "models/Forest.hpp"
#include "models/Tree.hpp"
#include "models/TreeCondition.hpp"
#include "models/TreeResponse.hpp"
#include "models/TrainingSpecGLDA.hpp"
#include "models/TrainingSpecUGLDA.hpp"
#include "utils/Macros.hpp"

using namespace pptree;
using namespace pptree::pp;
using namespace pptree::stats;
using namespace pptree::types;

static Projector as_projector(std::vector<Feature> v) {
  Eigen::Map<Projector> p(v.data(), v.size());
  return p;
}

// ---------------------------------------------------------------------------
// oob_indices
// ---------------------------------------------------------------------------

TEST(BootstrapTreeOobIndices, ComplementOfSampleIndices) {
  // sample_indices = {0, 1, 2}, n_total = 5  =>  OOB = {3, 4}
  BootstrapTree bt(
    TreeResponse::make(1),
    TrainingSpecGLDA::make(0.0f),
    std::vector<int>{ 0, 1, 2 });

  auto oob = bt.oob_indices(5);

  ASSERT_EQ(oob.size(), 2u);
  ASSERT_EQ(oob[0], 3);
  ASSERT_EQ(oob[1], 4);
}

TEST(BootstrapTreeOobIndices, EmptyWhenAllInBag) {
  BootstrapTree bt(
    TreeResponse::make(1),
    TrainingSpecGLDA::make(0.0f),
    std::vector<int>{ 0, 1, 2, 3 });

  auto oob = bt.oob_indices(4);
  ASSERT_TRUE(oob.empty());
}

TEST(BootstrapTreeOobIndices, AllOobWhenNoneInBag) {
  BootstrapTree bt(
    TreeResponse::make(1),
    TrainingSpecGLDA::make(0.0f),
    std::vector<int>{});

  auto oob = bt.oob_indices(3);

  ASSERT_EQ(oob.size(), 3u);
  ASSERT_EQ(oob[0], 0);
  ASSERT_EQ(oob[1], 1);
  ASSERT_EQ(oob[2], 2);
}

TEST(BootstrapTreeOobIndices, DuplicatesInSampleCountedOnce) {
  // With duplicates {0, 0, 1} the in-bag set is {0, 1}, OOB = {2, 3}
  BootstrapTree bt(
    TreeResponse::make(1),
    TrainingSpecGLDA::make(0.0f),
    std::vector<int>{ 0, 0, 1 });

  auto oob = bt.oob_indices(4);

  ASSERT_EQ(oob.size(), 2u);
  ASSERT_EQ(oob[0], 2);
  ASSERT_EQ(oob[1], 3);
}

// ---------------------------------------------------------------------------
// predict_oob
// ---------------------------------------------------------------------------

TEST(BootstrapTreePredictOob, MatchesRowwisePredict) {
  // Tree splits at 5.0 on x[0]: rows with x[0] < 5 -> 0, else -> 1
  auto condition = TreeCondition::make(
    as_projector({ 1.0f, 0.0f }),
    5.0f,
    TreeResponse::make(0),
    TreeResponse::make(1),
    nullptr,
    { 0, 1 },
    0.9f);

  BootstrapTree bt(
    std::move(condition),
    TrainingSpecGLDA::make(0.0f),
    std::vector<int>{ 0, 1, 4, 5 });

  FeatureMatrix x = DATA(Feature, 6,
    0.0f, 0.5f,
    0.1f, 0.3f,
    0.2f, 0.7f,
    9.8f, 0.4f,
    9.9f, 0.6f,
    9.7f, 0.2f);

  std::vector<int> oob_idx = { 2, 3 };
  ResponseVector preds = bt.predict_oob(x, oob_idx);

  ASSERT_EQ(preds.size(), 2);
  EXPECT_EQ(preds(0), 0) << "Row 2 has x[0]=0.2 < 5";
  EXPECT_EQ(preds(1), 1) << "Row 3 has x[0]=9.8 > 5";
}

TEST(BootstrapTreePredictOob, EmptyIndicesReturnsEmptyVector) {
  BootstrapTree bt(
    TreeResponse::make(1),
    TrainingSpecGLDA::make(0.0f),
    std::vector<int>{ 0, 1 });

  FeatureMatrix x(4, 2);
  x << 0, 0, 1, 1, 9, 9, 8, 8;

  ResponseVector preds = bt.predict_oob(x, std::vector<int>{});

  ASSERT_EQ(preds.size(), 0);
}

// ---------------------------------------------------------------------------
// pp_index_value stored during training
// ---------------------------------------------------------------------------

// Visitor that collects all pp_index_value fields from a tree.
struct IndexCollector : public TreeNodeVisitor {
  std::vector<Feature> values;

  void visit(const TreeCondition& node) override {
    values.push_back(node.pp_index_value);
    node.lower->accept(*this);
    node.upper->accept(*this);
  }

  void visit(const TreeResponse& /*node*/) override {
  }
};

TEST(PpIndexValue, StoredAfterTraining) {
  // Two well-separated groups: col 0 discriminates.
  FeatureMatrix x = DATA(Feature, 6,
    0.0f, 0.5f,
    0.1f, 0.3f,
    0.2f, 0.7f,
    9.8f, 0.4f,
    9.9f, 0.6f,
    9.7f, 0.2f);

  ResponseVector y = DATA(Response, 6,
    0, 0, 0, 1, 1, 1);

  stats::RNG rng(42, 0);
  Tree tree = Tree::train(TrainingSpecGLDA(0.0f), x, y, rng);

  IndexCollector collector;
  tree.root->accept(collector);

  ASSERT_FALSE(collector.values.empty());

  bool any_nonzero = false;

  for (Feature v : collector.values) {
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
  auto condition = TreeCondition::make(
    as_projector({ 1.0f, 0.0f }),
    5.0f,
    TreeResponse::make(0),
    TreeResponse::make(1),
    nullptr,
    { 0, 1 },
    0.9f);

  Tree tree(std::move(condition));

  FeatureVector vi2 = variable_importance_projections(tree, 2);

  ASSERT_NEAR(vi2(0), 0.5f, 1e-5f);
  ASSERT_NEAR(vi2(1), 0.0f, 1e-5f);
}

TEST(VariableImportance2, TreeOverloadTwoNodes) {
  // Root:  projector=[1,0], G_s=3
  //   -> lower leaf
  //   -> inner: projector=[0,1], G_s=2
  //        -> leaf, leaf
  //
  // VI2[0] = |1|/3 + |0|/2 = 1/3
  // VI2[1] = |0|/3 + |1|/2 = 1/2
  auto inner = TreeCondition::make(
    as_projector({ 0.0f, 1.0f }),
    3.0f,
    TreeResponse::make(1),
    TreeResponse::make(2),
    nullptr,
    { 1, 2 },
    0.6f);

  auto root = TreeCondition::make(
    as_projector({ 1.0f, 0.0f }),
    5.0f,
    TreeResponse::make(0),
    std::move(inner),
    nullptr,
    { 0, 1, 2 },
    0.8f);

  Tree tree(std::move(root));

  FeatureVector vi2 = variable_importance_projections(tree, 2);

  ASSERT_NEAR(vi2(0), 1.0f / 3.0f, 1e-5f);
  ASSERT_NEAR(vi2(1), 1.0f / 2.0f, 1e-5f);
}

TEST(VariableImportance2, TreeOverloadWithScale) {
  // projector=[0.5, 0.2], G_s=2
  // scale=[2.0, 3.0]
  // VI2[0] = |0.5| * 2.0 / 2 = 0.5
  // VI2[1] = |0.2| * 3.0 / 2 = 0.3
  auto condition = TreeCondition::make(
    as_projector({ 0.5f, 0.2f }),
    5.0f,
    TreeResponse::make(0),
    TreeResponse::make(1),
    nullptr,
    { 0, 1 },
    0.9f);

  Tree tree(std::move(condition));

  FeatureVector scale(2);
  scale << 2.0f, 3.0f;

  FeatureVector vi2 = variable_importance_projections(tree, 2, &scale);

  ASSERT_NEAR(vi2(0), 0.5f, 1e-5f);
  ASSERT_NEAR(vi2(1), 0.3f, 1e-5f);
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

  FeatureVector vi2 = variable_importance_projections(forest, 2);

  ASSERT_NEAR(vi2(0), 0.5f, 1e-5f);
  ASSERT_NEAR(vi2(1), 0.0f, 1e-5f);
}

TEST(VariableImportance2, HandBuiltTwoNodeTree) {
  // Root:  projector=[1,0], G_s=3, pp_index_value=0.8
  //   -> lower leaf
  //   -> inner: projector=[0,1], G_s=2, pp_index_value=0.6
  //        -> leaf, leaf
  //
  // VI2[0] = (1/B) * (|1|/3 + |0|/2) = 1/3
  // VI2[1] = (1/B) * (|0|/3 + |1|/2) = 1/2

  auto inner = TreeCondition::make(
    as_projector({ 0.0f, 1.0f }),
    3.0f,
    TreeResponse::make(1),
    TreeResponse::make(2),
    nullptr,
    { 1, 2 },
    0.6f);

  auto root = TreeCondition::make(
    as_projector({ 1.0f, 0.0f }),
    5.0f,
    TreeResponse::make(0),
    std::move(inner),
    nullptr,
    { 0, 1, 2 },
    0.8f);

  Forest forest;
  forest.add_tree(
    std::make_unique<BootstrapTree>(
      std::move(root),
      TrainingSpecGLDA::make(0.0f),
      std::vector<int>{ 0, 1, 2, 3 }));

  FeatureVector vi2 = variable_importance_projections(forest, 2);

  ASSERT_NEAR(vi2(0), 1.0f / 3.0f, 1e-5f);
  ASSERT_NEAR(vi2(1), 1.0f / 2.0f, 1e-5f);
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
      std::vector<int>{ 0, 1, 2, 3 }));  // all in-bag

  FeatureVector vi3 = variable_importance_weighted_projections(forest, x, y);

  ASSERT_NEAR(vi3(0), 0.9f, 1e-5f);
  ASSERT_NEAR(vi3(1), 0.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// VI1 — discriminating variable gets highest importance
// ---------------------------------------------------------------------------

TEST(VariableImportance1, DiscriminatingVariableHighestImportance) {
  // Col 0 perfectly separates the two groups.
  // Col 1 is pure noise (same values in both groups).
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

  Forest forest = Forest::train(
    TrainingSpecGLDA(0.0f),
    x, y,
    10,    // 10 trees
    42);

  FeatureVector vi1 = variable_importance_permuted(forest, x, y, 42);

  ASSERT_GT(vi1(0), vi1(1))
    << "Expected discriminating variable (col 0) to have higher VI1 than noise (col 1). "
    << "VI1[0]=" << vi1(0) << ", VI1[1]=" << vi1(1);
}
