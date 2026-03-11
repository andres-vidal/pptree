#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "models/Visualization.hpp"
#include "models/Tree.hpp"
#include "models/TreeCondition.hpp"
#include "models/TreeResponse.hpp"
#include "models/TrainingSpecGLDA.hpp"
#include "stats/Simulation.hpp"
#include "stats/Stats.hpp"

using namespace pptree;
using namespace pptree::pp;
using namespace pptree::stats;
using namespace pptree::types;

static Projector make_proj(std::vector<Feature> v) {
  Eigen::Map<Projector> proj(v.data(), static_cast<Eigen::Index>(v.size()));
  return proj;
}

// Build a 3-class iris-like tree:
//   Root: proj=[0.7, 0.3, 0.5, 0.1], thr=1.5
//     Left leaf: class 0
//     Right condition: proj=[0.2, 0.8, 0.1, 0.4], thr=2.0
//       Left leaf: class 1
//       Right leaf: class 2
static Tree make_test_tree() {
  auto leaf0 = TreeResponse::make(0);
  auto leaf1 = TreeResponse::make(1);
  auto leaf2 = TreeResponse::make(2);

  auto right_cond = TreeCondition::make(
    make_proj({ 0.2f, 0.8f, 0.1f, 0.4f }),
    2.0f,
    std::move(leaf1),
    std::move(leaf2),
    nullptr,
    { 1, 2 });

  auto root = TreeCondition::make(
    make_proj({ 0.7f, 0.3f, 0.5f, 0.1f }),
    1.5f,
    std::move(leaf0),
    std::move(right_cond),
    nullptr,
    { 0, 1, 2 });

  return Tree(std::move(root), TrainingSpecGLDA::make(0.5f));
}

// Make a deeper tree (depth 4) to stress-test memory management
static Tree make_deep_tree() {
  auto make_subtree = [](int left_class, int right_class,
                          std::vector<Feature> proj, Feature thr) {
      return TreeCondition::make(
        make_proj(proj), thr,
        TreeResponse::make(left_class),
        TreeResponse::make(right_class),
        nullptr,
        { left_class, right_class });
    };

  auto left_subtree = make_subtree(0, 1, { 0.6f, 0.4f }, 1.0f);
  auto right_subtree = make_subtree(2, 3, { 0.3f, 0.7f }, 2.0f);

  auto mid_left = TreeCondition::make(
    make_proj({ 0.5f, 0.5f }), 1.5f,
    std::move(left_subtree),
    TreeResponse::make(1),
    nullptr,
    { 0, 1 });

  auto mid_right = TreeCondition::make(
    make_proj({ 0.8f, 0.2f }), 2.5f,
    TreeResponse::make(2),
    std::move(right_subtree),
    nullptr,
    { 2, 3 });

  auto root = TreeCondition::make(
    make_proj({ 0.4f, 0.6f }), 1.8f,
    std::move(mid_left),
    std::move(mid_right),
    nullptr,
    { 0, 1, 2, 3 });

  return Tree(std::move(root), TrainingSpecGLDA::make(0.5f));
}

class VisualizationTest : public ::testing::Test {
protected:
  FeatureMatrix x;
  ResponseVector y;
  Tree tree = make_test_tree();

  void SetUp() override {
    // 30 observations, 4 features
    x.resize(30, 4);
    y.resize(30);

    RNG rng(42);

    for (int i = 0; i < 30; ++i) {
      for (int j = 0; j < 4; ++j) {
        x(i, j) = static_cast<Feature>(i * 0.1f + j * 0.5f + (rng() % 100) * 0.01f);
      }

      y(i) = i % 3;
    }
  }
};

TEST_F(VisualizationTest, NodeDataVisitorCollectsAllNodes) {
  NodeDataVisitor visitor(x, y);
  tree.root->accept(visitor);

  // Tree has 5 nodes: root, leaf0, right_cond, leaf1, leaf2
  ASSERT_EQ(visitor.nodes.size(), 5u);

  // Root is internal
  EXPECT_FALSE(visitor.nodes[0].is_leaf);
  EXPECT_EQ(visitor.nodes[0].depth, 0);
  EXPECT_EQ(visitor.nodes[0].projected_values.size(), 30u);
  EXPECT_EQ(visitor.nodes[0].classes.size(), 30u);

  // Check that at least one leaf exists
  bool found_leaf = false;

  for (const auto& nd : visitor.nodes) {
    if (nd.is_leaf) {
      found_leaf = true;
      EXPECT_TRUE(nd.projected_values.empty());
    }
  }

  EXPECT_TRUE(found_leaf);
}

TEST_F(VisualizationTest, BoundaryVisitorCollectsSegments) {
  BoundaryVisitor visitor(
    0, 1, {},
    -1.0f, 5.0f, -1.0f, 5.0f);

  tree.root->accept(visitor);

  // Should have at least 1 boundary segment (one per internal node that clips)
  EXPECT_GT(visitor.segments.size(), 0u);

  for (const auto& seg : visitor.segments) {
    EXPECT_GE(seg.depth, 0);
  }
}

TEST_F(VisualizationTest, BoundaryVisitorWithFixedVars) {
  std::vector<std::pair<int, Feature>> fixed = { { 2, 1.5f }, { 3, 0.5f } };
  BoundaryVisitor visitor(
    0, 1, fixed,
    -1.0f, 5.0f, -1.0f, 5.0f);

  tree.root->accept(visitor);

  // Should still produce segments
  EXPECT_GT(visitor.segments.size(), 0u);
}

TEST_F(VisualizationTest, RegionVisitorCollectsPolygons) {
  RegionVisitor visitor(
    0, 1, {},
    -1.0f, 5.0f, -1.0f, 5.0f);

  tree.root->accept(visitor);

  // Should have one region per leaf (3 leaves)
  EXPECT_EQ(visitor.regions.size(), 3u);

  for (const auto& region : visitor.regions) {
    // Each region should be a polygon with at least 3 vertices
    EXPECT_GE(region.vertices.size(), 3u);
    // Predicted class should be valid
    EXPECT_GE(region.predicted_class, 0);
    EXPECT_LE(region.predicted_class, 2);
  }
}

TEST_F(VisualizationTest, RegionVisitorWithFixedVars) {
  std::vector<std::pair<int, Feature>> fixed = { { 2, 1.5f }, { 3, 0.5f } };
  RegionVisitor visitor(
    0, 1, fixed,
    -1.0f, 5.0f, -1.0f, 5.0f);

  tree.root->accept(visitor);

  EXPECT_EQ(visitor.regions.size(), 3u);
}

TEST_F(VisualizationTest, ClipPolygonHalfspaceBasic) {
  // Unit square
  Polygon square = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };

  // Clip with x < 0.5
  FeatureVector normal(2);
  normal << 1.0f, 0.0f;
  Polygon clipped = clip_polygon_halfspace(square, normal, 0.5f, true);

  EXPECT_EQ(clipped.size(), 4u);

  for (const auto& v : clipped) {
    EXPECT_LT(v.first, 0.5f + 1e-6f);
  }
}

TEST_F(VisualizationTest, ClipPolygonHalfspaceEmpty) {
  Polygon square = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };

  // Clip with x < -1 (nothing inside)
  FeatureVector normal(2);
  normal << 1.0f, 0.0f;
  Polygon clipped = clip_polygon_halfspace(square, normal, -1.0f, true);

  EXPECT_TRUE(clipped.empty());
}

TEST_F(VisualizationTest, ComputeTreeLayoutProducesCorrectCounts) {
  TreeLayout layout = compute_tree_layout(*tree.root);

  // 5 nodes, 4 edges (each internal node has 2 children edges)
  EXPECT_EQ(layout.nodes.size(), 5u);
  EXPECT_EQ(layout.edges.size(), 4u);

  // Check that nodes have unique positions
  for (size_t i = 0; i < layout.nodes.size(); ++i) {
    for (size_t j = i + 1; j < layout.nodes.size(); ++j) {
      bool same_pos = (layout.nodes[i].x == layout.nodes[j].x &&
                        layout.nodes[i].y == layout.nodes[j].y);
      EXPECT_FALSE(same_pos) << "Nodes " << i << " and " << j << " overlap";
    }
  }
}

TEST_F(VisualizationTest, ComputeTreeLayoutLeafAndInternalFlags) {
  TreeLayout layout = compute_tree_layout(*tree.root);

  int n_internal = 0;
  int n_leaf = 0;

  for (const auto& node : layout.nodes) {
    if (node.is_leaf) n_leaf++;
    else n_internal++;
  }

  EXPECT_EQ(n_internal, 2);
  EXPECT_EQ(n_leaf, 3);
}

TEST_F(VisualizationTest, ComputeTreeLayoutEdgesHaveLabels) {
  TreeLayout layout = compute_tree_layout(*tree.root);

  for (const auto& edge : layout.edges) {
    EXPECT_FALSE(edge.label.empty());
  }
}

// Deep tree tests stress memory management under ASan
TEST(VisualizationDeepTree, NodeDataVisitor) {
  Tree deep = make_deep_tree();

  FeatureMatrix x(20, 2);
  ResponseVector y(20);

  for (int i = 0; i < 20; ++i) {
    x(i, 0) = static_cast<Feature>(i * 0.2f);
    x(i, 1) = static_cast<Feature>(i * 0.3f);
    y(i) = i % 4;
  }

  NodeDataVisitor visitor(x, y);
  deep.root->accept(visitor);

  // 11 nodes: 5 internal + 6 leaves
  EXPECT_EQ(visitor.nodes.size(), 11u);
}

TEST(VisualizationDeepTree, BoundaryVisitor) {
  Tree deep = make_deep_tree();

  BoundaryVisitor visitor(0, 1, {}, -1.0f, 10.0f, -1.0f, 10.0f);
  deep.root->accept(visitor);

  EXPECT_GT(visitor.segments.size(), 0u);
}

TEST(VisualizationDeepTree, RegionVisitor) {
  Tree deep = make_deep_tree();

  RegionVisitor visitor(0, 1, {}, -1.0f, 10.0f, -1.0f, 10.0f);
  deep.root->accept(visitor);

  // Should have one region per leaf (6 leaves)
  EXPECT_EQ(visitor.regions.size(), 6u);
}

TEST(VisualizationDeepTree, ComputeTreeLayout) {
  Tree deep = make_deep_tree();

  TreeLayout layout = compute_tree_layout(*deep.root);

  // 11 nodes, 10 edges
  EXPECT_EQ(layout.nodes.size(), 11u);
  EXPECT_EQ(layout.edges.size(), 10u);
}

// Test repeated visitor creation/destruction (stress allocator)
TEST(VisualizationStress, RepeatedVisitorCalls) {
  Tree tree = make_test_tree();

  for (int iter = 0; iter < 50; ++iter) {
    {
      BoundaryVisitor bv(0, 1, {}, -1.0f, 5.0f, -1.0f, 5.0f);
      tree.root->accept(bv);
      ASSERT_GT(bv.segments.size(), 0u);
    }

    {
      RegionVisitor rv(0, 1, {}, -1.0f, 5.0f, -1.0f, 5.0f);
      tree.root->accept(rv);
      ASSERT_EQ(rv.regions.size(), 3u);
    }

    {
      TreeLayout layout = compute_tree_layout(*tree.root);
      ASSERT_EQ(layout.nodes.size(), 5u);
    }
  }
}

// Test with many different variable pairs (like pairwise plot does)
TEST_F(VisualizationTest, PairwiseBoundaryAndRegions) {
  int p = 4;

  for (int i = 0; i < p; ++i) {
    for (int j = i + 1; j < p; ++j) {
      std::vector<std::pair<int, Feature>> fixed;

      for (int k = 0; k < p; ++k) {
        if (k != i && k != j) {
          fixed.push_back({ k, 1.0f });
        }
      }

      BoundaryVisitor bv(i, j, fixed, -1.0f, 5.0f, -1.0f, 5.0f);
      tree.root->accept(bv);
      EXPECT_GT(bv.segments.size(), 0u)
        << "No segments for pair (" << i << ", " << j << ")";

      RegionVisitor rv(i, j, fixed, -1.0f, 5.0f, -1.0f, 5.0f);
      tree.root->accept(rv);
      EXPECT_EQ(rv.regions.size(), 3u)
        << "Wrong region count for pair (" << i << ", " << j << ")";
    }
  }
}
