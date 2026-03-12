#pragma once

/**
 * @file Visualization.hpp
 * @brief Tree-analysis visitors and geometric utilities for rendering
 *        projection-pursuit tree decision boundaries and structure diagrams.
 *
 * @section arch Architecture
 *
 * C++ (this module) handles geometry and tree traversal:
 *
 * - **NodeDataVisitor** — Routes observations through the tree, collects
 *   per-node projected values and class labels for histogram rendering in
 *   the tree structure diagram.
 *
 * - **BoundaryVisitor** — Projects each split's decision boundary line
 *   into a 2D feature plane and clips it to the visible bounding box and
 *   all ancestor half-space constraints using parametric line clipping.
 *
 * - **RegionVisitor** — Computes convex decision region polygons for each
 *   leaf via Sutherland–Hodgman polygon clipping against ancestor
 *   half-spaces.  Produces one polygon per reachable leaf.
 *
 * - **compute_tree_layout** — Positions tree nodes on a 2D canvas with a
 *   recursive left-aligned layout algorithm.
 *
 * R (plot.R) handles rendering via ggplot2:
 *   Translates visitor output into ggplot2 layers (geom_polygon,
 *   geom_segment, geom_rect) and assembles composite layouts (mosaic,
 *   pairwise facets, tree diagram).
 *
 * @section proj 2D Projection
 *
 * When the feature space has p > 2 variables, the visualization selects
 * two variables (var_i, var_j) for the display axes and holds the
 * remaining p−2 variables at fixed values (typically medians).  Each
 * split's p-dimensional projector and threshold are reduced to 2D via
 * project_2d() and adjust_threshold(), preserving the boundary geometry
 * in the chosen 2D slice.
 *
 * @section protocol Visitor Protocol
 *
 * All visitors inherit from TreeNodeVisitor and implement visit() for
 * TreeCondition (internal) and TreeResponse (leaf) nodes.  Traversal
 * is initiated by calling tree.root->accept(visitor).  Results
 * accumulate in public member vectors (nodes, segments, regions) that
 * the R layer reads via Rcpp exports in main.cpp.
 *
 * @section safety Thread Safety
 *
 * Visitors are single-use, single-threaded.  Create a fresh visitor for
 * each traversal.  Multiple visitors may run concurrently on the same
 * (immutable) tree.
 */

#include "models/TreeNodeVisitor.hpp"
#include "models/TreeCondition.hpp"
#include "models/TreeResponse.hpp"
#include "utils/Types.hpp"

#include <vector>
#include <utility>
#include <string>

namespace ppforest2 {
  // ===================================================================
  // Node Data Collection
  // ===================================================================

  /**
   * @brief Per-node data collected by routing observations through the tree.
   *
   * For internal (condition) nodes:
   *   - @c projected_values: dot product of each reaching observation with
   *     the node's projector vector.
   *   - @c classes: class label (1-indexed response) of each reaching
   *     observation, parallel to @c projected_values.
   *   - @c projector, @c threshold: the split parameters (copied for
   *     convenience so the R side can render histograms without re-accessing
   *     the tree).
   *
   * For leaf (response) nodes:
   *   - @c projected_values is empty.
   *   - @c classes contains labels of observations reaching the leaf.
   *   - @c value is the predicted class (0-indexed response).
   */
  struct NodeData {
    bool is_leaf;
    int depth;
    types::FeatureVector projector;
    types::Feature threshold;
    types::Response value;
    std::vector<types::Feature> projected_values;
    std::vector<types::Response> classes;
  };

  /**
   * @brief Visitor that routes observations through the tree and collects
   *        per-node projection data for histogram rendering.
   *
   * Usage:
   * @code
   *   NodeDataVisitor visitor(x, y);
   *   tree.root->accept(visitor);
   *   // visitor.nodes is a pre-order vector of NodeData
   * @endcode
   *
   * At each split node the visitor projects all reaching observations onto
   * the node's projector, records the projected values and class labels,
   * partitions observation indices by the threshold, and recurses into the
   * lower and upper children.  Nodes are accumulated in **pre-order**
   * (parent before children, left before right).
   *
   * The observation routing respects the tree structure exactly: an
   * observation reaching a split goes to the lower child if its projected
   * value < threshold, otherwise to the upper child.
   */
  struct NodeDataVisitor : public TreeNodeVisitor {
    const types::FeatureMatrix& x;   ///< Full observation matrix (n × p).
    const types::ResponseVector& y;  ///< Full response vector (n).
    std::vector<int> indices;        ///< Indices of observations reaching the current node.
    int depth;                       ///< Current depth in the traversal.
    std::vector<NodeData> nodes;     ///< Collected node data (pre-order).

    NodeDataVisitor(
      const types::FeatureMatrix&  x,
      const types::ResponseVector& y);

    void visit(const TreeCondition& node) override;
    void visit(const TreeResponse& node) override;
  };

  // ===================================================================
  // Decision Boundary and Region Types
  // ===================================================================

  /**
   * @brief A half-space constraint derived from an ancestor split.
   *
   * Represents the constraint  projector^T x < threshold  (if @c is_lower)
   * or  projector^T x >= threshold  (if !is_lower).  Stored in the full
   * p-dimensional space; projected to 2D when needed by the visitors.
   *
   * As BoundaryVisitor and RegionVisitor traverse the tree, they
   * accumulate a stack of HalfSpace constraints — one per ancestor
   * split on the path from root to the current node.  These constraints
   * define the convex region in which the current node's boundary is
   * active (BoundaryVisitor) or the leaf's decision region (RegionVisitor).
   *
   * @see BoundaryVisitor, RegionVisitor
   */
  struct HalfSpace {
    types::FeatureVector projector;
    types::Feature threshold;
    bool is_lower;  ///< true = lower child side (projected value < threshold).
  };

  /**
   * @brief A clipped decision boundary line segment in 2D feature space.
   *
   * Each segment corresponds to one split node's boundary line, clipped to
   * the visible bounding box and to all ancestor half-space constraints.
   * The @c depth field records the tree depth for potential styling
   * (e.g., thinner lines at deeper levels).
   *
   * Produced by BoundaryVisitor; consumed by the R layer as a data frame
   * of (x_start, y_start, x_end, y_end, depth) rows, rendered with
   * ggplot2::geom_segment().
   *
   * @see BoundaryVisitor, clip_boundary_2d
   */
  struct BoundarySegment {
    types::Feature x_start, y_start, x_end, y_end;
    int depth;
  };

  /**
   * @brief A convex decision region polygon in 2D with its predicted class.
   *
   * Each leaf in the tree produces one region: the bounding box clipped by
   * all ancestor half-space constraints (Sutherland–Hodgman algorithm).
   * Vertices are ordered for polygon rendering (either CW or CCW, consistent
   * with the initial bounding box winding).
   *
   * The union of all leaf regions tiles the entire bounding box — every
   * point in the bbox belongs to exactly one leaf's region.  This
   * guarantees that region polygons fully cover the plot area when the
   * bounding box matches the visible coordinate range.
   *
   * Produced by RegionVisitor; consumed by the R layer via build_region_df()
   * and rendered with ggplot2::geom_polygon().
   *
   * @see RegionVisitor, clip_polygon_halfspace
   */
  struct RegionPolygon {
    std::vector<std::pair<types::Feature, types::Feature>> vertices;
    types::Response predicted_class;  ///< 0-indexed class label from the leaf.
  };

  // ===================================================================
  // 2D Projection Helpers
  // ===================================================================

  /**
   * @brief Extract a 2D sub-projection from a full p-dimensional projector.
   *
   * Returns a 2-vector containing the components at indices @p var_i and
   * @p var_j.  Used to reduce a p-dimensional split to the 2D plane being
   * visualized.
   *
   * @param full_proj  Full projector vector (p).
   * @param var_i      Index of the x-axis variable (0-based).
   * @param var_j      Index of the y-axis variable (0-based).
   * @return           2D projection vector [full_proj(var_i), full_proj(var_j)].
   */
  types::FeatureVector project_2d(
    const types::FeatureVector& full_proj,
    int var_i, int var_j);

  /**
   * @brief Adjust a split threshold by subtracting contributions of fixed variables.
   *
   * When projecting from p dimensions to 2D, the remaining (p − 2) variables
   * are held at fixed values (typically medians).  The effective 2D threshold
   * is:  t' = t − Σ_{k ∈ fixed} projector_k × value_k
   *
   * @param full_proj   Full projector vector (p).
   * @param thr         Original p-dimensional threshold.
   * @param fixed_vars  Pairs of (variable index, fixed value) for non-displayed variables.
   * @return            Adjusted 2D threshold.
   */
  types::Feature adjust_threshold(
    const types::FeatureVector&                        full_proj,
    types::Feature                                     thr,
    const std::vector<std::pair<int, types::Feature>>& fixed_vars);

  // ===================================================================
  // Parametric Line Clipping
  // ===================================================================

  /**
   * @brief Clip a parametric interval [u_min, u_max] to a 1D range.
   *
   * Given a parametric line  x(u) = origin + u × direction, restricts the
   * interval [u_min, u_max] so that x(u) ∈ [range_min, range_max].
   *
   * @param origin     Starting value.
   * @param direction  Rate of change.
   * @param range_min  Lower bound of the range.
   * @param range_max  Upper bound of the range.
   * @param[in,out] u_min  Current lower parameter bound (tightened in place).
   * @param[in,out] u_max  Current upper parameter bound (tightened in place).
   * @return           true if a valid interval remains (u_min < u_max).
   */
  bool clip_param_to_range(
    types::Feature origin, types::Feature direction,
    types::Feature range_min, types::Feature range_max,
    types::Feature& u_min, types::Feature& u_max);

  /**
   * @brief Clip a 2D decision boundary line to the visible rectangle and
   *        all ancestor half-space constraints.
   *
   * The boundary of a split with 2D projector @p a and threshold @p threshold
   * is the line  a^T x = threshold.  This function parametrizes the line,
   * clips to the bounding box [x_min, x_max] × [y_min, y_max], then clips
   * against each ancestor constraint.  If any visible portion remains, writes
   * the endpoints into @p segment and returns true.
   *
   * Algorithm (parametric line clipping):
   *   1. Compute direction D = (-a1, a0) tangent to the boundary line.
   *   2. Find a base point P0 on the line: a^T P0 = threshold.
   *   3. Parametrize as P(u) = P0 + u·D with u ∈ (-∞, +∞).
   *   4. Clip [u_min, u_max] to the bounding box (clip_param_to_range).
   *   5. For each ancestor constraint a_c^T x ⋛ t_c, compute the
   *      intersection parameter and tighten the interval.
   *   6. If u_min < u_max, the segment P(u_min)→P(u_max) is visible.
   *
   * @param a            2D projection of the split's projector.
   * @param threshold    Adjusted 2D threshold.
   * @param constraints  Ancestor half-space constraints (already in 2D).
   * @param x_min, x_max, y_min, y_max  Visible bounding box.
   * @param[out] segment  Output segment (valid only if function returns true).
   * @param depth        Tree depth of this split (stored in the segment).
   * @return             true if a visible segment remains after clipping.
   *
   * @see clip_param_to_range, BoundaryVisitor
   */
  bool clip_boundary_2d(
    const types::FeatureVector& a,
    types::Feature threshold,
    const std::vector<HalfSpace>& constraints,
    types::Feature x_min, types::Feature x_max,
    types::Feature y_min, types::Feature y_max,
    BoundarySegment& segment,
    int depth);

  // ===================================================================
  // Polygon Clipping (Sutherland–Hodgman)
  // ===================================================================

  /// A polygon represented as an ordered list of (x, y) vertex pairs.
  using Polygon = std::vector<std::pair<types::Feature, types::Feature>>;

  /**
   * @brief Clip a convex polygon against a single half-space.
   *
   * Implements one pass of the Sutherland–Hodgman algorithm.  The half-space
   * is defined by:
   *   - @c is_lower = true:  normal^T x < threshold  (strict)
   *   - @c is_lower = false: normal^T x >= threshold
   *
   * Vertices are processed in order; edges crossing the boundary are split
   * at the intersection point.  The result is a (possibly empty) polygon
   * fully contained in the half-space.
   *
   * For each edge (curr → next), exactly one of four cases applies:
   *   - Both inside:  emit next.
   *   - Leaving (curr inside, next outside):  emit intersection.
   *   - Entering (curr outside, next inside): emit intersection + next.
   *   - Both outside: emit nothing.
   *
   * Convexity is preserved: clipping a convex polygon against a
   * half-space always produces a convex polygon (or empty).
   *
   * Complexity: O(n) where n = number of polygon vertices.
   *
   * @param polygon    Input polygon (ordered vertices).
   * @param normal     2D normal vector of the half-space boundary.
   * @param threshold  Offset of the half-space boundary.
   * @param is_lower   Which side of the boundary to keep.
   * @return           Clipped polygon (empty if fully outside).
   *
   * @see RegionVisitor
   */
  Polygon clip_polygon_halfspace(
    const Polygon&              polygon,
    const types::FeatureVector& normal,
    types::Feature              threshold,
    bool                        is_lower);

  // ===================================================================
  // Boundary Visitor
  // ===================================================================

  /**
   * @brief Visitor that collects and clips decision boundary line segments.
   *
   * Traverses the tree, projecting each split's boundary into a 2D plane
   * defined by two feature variables (var_i, var_j).  All other variables
   * are held at fixed values to reduce the split from p-D to 2D.
   *
   * Each boundary line is clipped to:
   *   1. The bounding box [x_min, x_max] × [y_min, y_max].
   *   2. All ancestor half-space constraints (so the line only appears in
   *      the region where it is actually the active decision boundary).
   *
   * Usage:
   * @code
   *   BoundaryVisitor visitor(0, 1, fixed, x_min, x_max, y_min, y_max);
   *   tree.root->accept(visitor);
   *   // visitor.segments contains all visible boundary segments
   * @endcode
   */
  struct BoundaryVisitor : public TreeNodeVisitor {
    int var_i, var_j;  ///< Indices of the two displayed feature variables.
    std::vector<std::pair<int, types::Feature>> fixed_vars;  ///< Fixed (index, value) pairs.
    types::Feature x_min, x_max, y_min, y_max;  ///< Visible bounding box.
    int depth;
    std::vector<HalfSpace> constraints;      ///< Accumulated ancestor constraints.
    std::vector<BoundarySegment> segments;    ///< Output: clipped boundary segments.

    BoundaryVisitor(
      int var_i, int var_j,
      const std::vector<std::pair<int, types::Feature>>& fixed_vars,
      types::Feature x_min, types::Feature x_max,
      types::Feature y_min, types::Feature y_max);

    void visit(const TreeCondition& node) override;
    void visit(const TreeResponse& node) override;
  };

  // ===================================================================
  // Region Visitor
  // ===================================================================

  /**
   * @brief Visitor that collects convex decision region polygons.
   *
   * At each leaf, starts with the full bounding box as a rectangle and
   * clips it against every ancestor split's half-space constraint
   * (projected to 2D via project_2d / adjust_threshold).  The result is
   * a convex polygon representing the leaf's decision region in the
   * (var_i, var_j) plane.
   *
   * Uses the Sutherland–Hodgman algorithm (clip_polygon_halfspace) for
   * each constraint.  If the polygon becomes empty (leaf unreachable in
   * the 2D slice), it is silently skipped.
   *
   * Usage:
   * @code
   *   RegionVisitor visitor(0, 1, fixed, x_min, x_max, y_min, y_max);
   *   tree.root->accept(visitor);
   *   // visitor.regions has one polygon per reachable leaf
   * @endcode
   */
  struct RegionVisitor : public TreeNodeVisitor {
    int var_i, var_j;  ///< Indices of the two displayed feature variables.
    std::vector<std::pair<int, types::Feature>> fixed_vars;  ///< Fixed (index, value) pairs.
    types::Feature x_min, x_max, y_min, y_max;  ///< Bounding box for the initial rectangle.
    std::vector<HalfSpace> constraints;  ///< Accumulated ancestor constraints.
    std::vector<RegionPolygon> regions;  ///< Output: one polygon per reachable leaf.

    RegionVisitor(
      int var_i, int var_j,
      const std::vector<std::pair<int, types::Feature>>& fixed_vars,
      types::Feature x_min, types::Feature x_max,
      types::Feature y_min, types::Feature y_max);

    void visit(const TreeCondition& node) override;
    void visit(const TreeResponse& node) override;
  };

  // ===================================================================
  // Tree Layout
  // ===================================================================

  /**
   * @brief Layout parameters for tree structure rendering.
   *
   * Controls spacing and dimensions used by compute_tree_layout().
   * Node dimensions must match the R-side constants (ppforest2_node_w, etc.)
   * so that histogram bars and leaf labels align with the rectangles.
   */
  struct LayoutParams {
    types::Feature y_spacing = 1.5f;   ///< Vertical distance between depth levels.
    types::Feature node_w    = 0.8f;   ///< Internal node width.
    types::Feature node_h    = 0.7f;   ///< Internal node height.
    types::Feature leaf_w    = 0.5f;   ///< Leaf node width.
    types::Feature leaf_h    = 0.3f;   ///< Leaf node height.
    types::Feature gap       = 0.2f;   ///< Horizontal gap between sibling subtrees.
  };

  /**
   * @brief A positioned tree node in the computed layout.
   */
  struct LayoutNode {
    types::Feature x, y;  ///< Center position of the node.
    bool is_leaf;
    int node_idx;         ///< Pre-order index (0-based).
  };

  /**
   * @brief An edge between two positioned nodes with a threshold label.
   */
  struct LayoutEdge {
    types::Feature from_x, from_y, to_x, to_y;
    std::string label;  ///< e.g. "< 1.50" or "≥ 1.50".
  };

  /**
   * @brief Complete tree layout: positioned nodes and labelled edges.
   */
  struct TreeLayout {
    std::vector<LayoutNode> nodes;
    std::vector<LayoutEdge> edges;
  };

  /**
   * @brief Compute a left-aligned tree layout for rendering.
   *
   * Recursively positions nodes on a 2D canvas:
   *   - Internal nodes are centered above their **left** child (left-aligned).
   *   - Leaf nodes are placed at the bottom of their subtree column.
   *   - y decreases with depth (root at top, leaves at bottom).
   *   - Sibling subtrees are separated by @c params.gap.
   *
   * Edges connect each parent's bottom center to each child's top center.
   * Each edge is labelled with the split direction and threshold value
   * (e.g. "< 1.50" for the lower child, "≥ 1.50" for the upper child).
   *
   * Nodes are accumulated in **pre-order** traversal (matching
   * NodeDataVisitor's order, so indices correspond).  This invariant is
   * critical: the R layer uses the pre-order index to match layout
   * positions with the node data from NodeDataVisitor.
   *
   * Algorithm (recursive bottom-up):
   *   1. For a leaf, place it at x_offset + leaf_w/2 and return its width.
   *   2. For an internal node, layout left and right subtrees side-by-side
   *      (with gap), then place the parent above the left child's center.
   *   3. Total subtree width = left_width + gap + right_width.
   *
   * @param root    Root node of the tree to layout.
   * @param params  Layout dimensions and spacing (default: LayoutParams()).
   * @return        TreeLayout with positioned nodes and edges.
   */
  TreeLayout compute_tree_layout(
    const TreeNode&     root,
    const LayoutParams& params = LayoutParams());
}
