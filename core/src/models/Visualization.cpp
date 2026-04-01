/**
 * @file Visualization.cpp
 * @brief Implementation of tree-analysis visitors and geometric
 *        utilities for visualization.
 *
 * @see Visualization.hpp for the public API and module overview.
 */

#include "models/Visualization.hpp"

#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace ppforest2::viz {
  /**
   * NodeDataVisitor
   *
   * Traverses the tree in pre-order, routing observations by the split
   * condition at each internal node.  At each condition node, every
   * reaching observation is projected onto the node's projector, and
   * the projected value is compared against the threshold to partition
   * observations into the lower and upper children.
   *
   * The result is a pre-order sequence of NodeData structs containing:
   *   - For internal nodes: projected values, group labels, projector,
   *     threshold (used for histogram rendering in the tree diagram).
   *   - For leaf nodes: group labels of reaching observations, and the
   *     predicted group (used for leaf labels and coloring).
   */
  NodeDataVisitor::NodeDataVisitor(types::FeatureMatrix const& x, types::ResponseVector const& y)
      : x(x)
      , y(y)
      , depth(0) {
    // Initially all observations are routed through the root.
    indices.resize(static_cast<std::size_t>(x.rows()));

    for (int i = 0; i < x.rows(); ++i) {
      indices[static_cast<std::size_t>(i)] = i;
    }
  }

  void NodeDataVisitor::visit(TreeCondition const& node) {
    // Collect projected values for all observations reaching this node.
    NodeData nd;
    nd.is_leaf   = false;
    nd.depth     = depth;
    nd.projector = node.projector;
    nd.threshold = node.threshold;
    nd.value     = 0;

    // Partition observations into lower/upper based on projected value.
    std::vector<int> lower_indices;
    std::vector<int> upper_indices;

    for (int idx : indices) {
      types::Feature proj_val = x.row(idx).dot(node.projector);
      nd.projected_values.push_back(proj_val);
      nd.groups.push_back(y(idx));

      if (proj_val < node.threshold) {
        lower_indices.push_back(idx);
      } else {
        upper_indices.push_back(idx);
      }
    }

    nodes.push_back(std::move(nd));

    // Save and swap indices for child traversal.
    auto saved_indices = std::move(indices);

    indices = std::move(lower_indices);
    depth++;
    node.lower->accept(*this);

    indices = std::move(upper_indices);
    node.upper->accept(*this);
    depth--;

    indices = std::move(saved_indices);
  }

  void NodeDataVisitor::visit(TreeResponse const& node) {
    NodeData nd;
    nd.is_leaf   = true;
    nd.depth     = depth;
    nd.threshold = 0;
    nd.value     = node.value;

    for (int idx : indices) {
      nd.groups.push_back(y(idx));
    }

    nodes.push_back(std::move(nd));
  }

  /**
   * 2D Projection Helpers
   *
   * When visualizing a p-dimensional tree in 2D, we select two variables
   * (var_i, var_j) for the axes and hold all others at fixed values
   * (typically medians).  Each split's projector and threshold are then
   * reduced to 2D by extracting the two relevant components and
   * subtracting the fixed variables' contributions from the threshold.
   */
  types::FeatureVector project_2d(types::FeatureVector const& full_proj, int var_i, int var_j) {
    types::FeatureVector proj2d(2);
    proj2d(0) = full_proj(var_i);
    proj2d(1) = full_proj(var_j);
    return proj2d;
  }

  types::Feature adjust_threshold(
      types::FeatureVector const& full_proj,
      types::Feature thr,
      std::vector<std::pair<int, types::Feature>> const& fixed_vars
  ) {
    // t' = t - Σ projector_k * value_k  for each fixed variable k
    for (auto const& fv : fixed_vars) {
      thr -= full_proj(fv.first) * fv.second;
    }

    return thr;
  }

  /**
   * Parametric Line Clipping
   *
   * Decision boundaries in 2D are lines of the form  a^T x = t.  We
   * parametrize each line as  P(u) = P0 + u * D  where D is tangent to
   * the line (perpendicular to 'a') and P0 is any point on the line.
   *
   * Clipping to a rectangle and ancestor half-spaces reduces to
   * tightening the interval [u_min, u_max].  If u_min >= u_max after
   * all constraints, the line is fully outside the visible region.
   */
  bool clip_param_to_range(
      types::Feature origin,
      types::Feature direction,
      types::Feature range_min,
      types::Feature range_max,
      types::Feature& u_min,
      types::Feature& u_max
  ) {
    types::Feature const eps = static_cast<types::Feature>(1e-12);

    if (std::abs(direction) < eps) {
      // Line is parallel to this axis — check if origin is in range.
      return origin >= range_min && origin <= range_max;
    }

    types::Feature u1 = (range_min - origin) / direction;
    types::Feature u2 = (range_max - origin) / direction;

    if (direction < 0)
      std::swap(u1, u2);

    u_min = std::max(u_min, u1);
    u_max = std::min(u_max, u2);

    return u_min < u_max;
  }

  bool clip_boundary_2d(
      types::FeatureVector const& a,
      types::Feature threshold,
      std::vector<HalfSpace> const& constraints,
      types::Feature x_min,
      types::Feature x_max,
      types::Feature y_min,
      types::Feature y_max,
      BoundarySegment& segment,
      int depth
  ) {
    types::Feature const eps = static_cast<types::Feature>(1e-12);

    // Direction tangent to the boundary line a^T x = threshold.
    // Perpendicular to a = (a0, a1) is D = (-a1, a0).
    types::Feature dx = -a(1);
    types::Feature dy = a(0);

    // Find a point P0 on the line.
    types::Feature px, py;

    if (std::abs(a(1)) > std::abs(a(0))) {
      px = 0;
      py = threshold / a(1);
    } else {
      px = threshold / a(0);
      py = 0;
    }

    // Start with a large parametric interval and clip.
    types::Feature u_min = static_cast<types::Feature>(-1e6);
    types::Feature u_max = static_cast<types::Feature>(1e6);

    // Clip to bounding box.
    if (!clip_param_to_range(px, dx, x_min, x_max, u_min, u_max))
      return false;

    if (!clip_param_to_range(py, dy, y_min, y_max, u_min, u_max))
      return false;

    // Clip against each ancestor half-space constraint.
    for (auto const& con : constraints) {
      types::Feature denom = con.projector(0) * dx + con.projector(1) * dy;

      if (std::abs(denom) < eps)
        continue;

      types::Feature u_intersect = (con.threshold - (con.projector(0) * px + con.projector(1) * py)) / denom;

      if (con.is_lower) {
        // Constraint: con.projector^T x < con.threshold
        if (denom > 0)
          u_max = std::min(u_max, u_intersect);
        else
          u_min = std::max(u_min, u_intersect);
      } else {
        // Constraint: con.projector^T x >= con.threshold
        if (denom > 0)
          u_min = std::max(u_min, u_intersect);
        else
          u_max = std::min(u_max, u_intersect);
      }

      if (u_min >= u_max)
        return false;
    }

    // Compute clipped endpoints.
    segment.x_start = px + u_min * dx;
    segment.y_start = py + u_min * dy;
    segment.x_end   = px + u_max * dx;
    segment.y_end   = py + u_max * dy;
    segment.depth   = depth;
    return true;
  }

  /**
   * Polygon Clipping (Sutherland–Hodgman)
   *
   * Used by RegionVisitor to compute decision region polygons.  Each
   * leaf's region starts as the full bounding box rectangle, then is
   * progressively clipped by each ancestor's half-space constraint.
   * Since all constraints are linear half-spaces and the initial shape
   * is convex, the result is always a convex polygon (or empty).
   *
   * The key property for visualization: the union of all leaf polygons
   * tiles the entire bounding box.  Every point in the bbox is
   * classified by the tree into exactly one leaf, so the corresponding
   * leaf's polygon contains that point.  This guarantees complete
   * background coverage in the plot when the bbox matches the visible
   * coordinate range (i.e., no white gaps between regions).
   *
   * Complexity: O(k × n) per leaf, where k = tree depth (number of
   * ancestor constraints) and n = max polygon vertices at any step.
   * Since k and n are both bounded by tree depth, this is effectively
   * O(depth²) per leaf.
   */

  Polygon clip_polygon_halfspace(
      Polygon const& polygon, types::FeatureVector const& normal, types::Feature threshold, bool is_lower
  ) {
    if (polygon.empty())
      return {}

      ;

    Polygon result;
    std::size_t n = polygon.size();

    for (std::size_t i = 0; i < n; ++i) {
      auto const& curr = polygon[i];
      auto const& next = polygon[(i + 1) % n];

      // Signed distance from the boundary: normal^T x - threshold
      types::Feature d_curr = normal(0) * curr.first + normal(1) * curr.second - threshold;
      types::Feature d_next = normal(0) * next.first + normal(1) * next.second - threshold;

      // Determine which side each vertex is on.
      //   is_lower (a^T x < t):  inside means d < 0
      //   !is_lower (a^T x >= t): inside means d >= 0
      bool curr_inside = is_lower ? (d_curr < 0) : (d_curr >= 0);
      bool next_inside = is_lower ? (d_next < 0) : (d_next >= 0);

      if (curr_inside && next_inside) {
        // Both inside — emit next vertex.
        result.push_back(next);
      } else if (curr_inside && !next_inside) {
        // Leaving — emit intersection point.
        types::Feature t_param = d_curr / (d_curr - d_next);
        result.push_back(
            {curr.first + t_param * (next.first - curr.first), curr.second + t_param * (next.second - curr.second)}
        );
      } else if (!curr_inside && next_inside) {
        // Entering — emit intersection point then next vertex.
        types::Feature t_param = d_curr / (d_curr - d_next);
        result.push_back(
            {curr.first + t_param * (next.first - curr.first), curr.second + t_param * (next.second - curr.second)}
        );
        result.push_back(next);
      }

      // Both outside — emit nothing.
    }

    return result;
  }

  /**
   * BoundaryVisitor
   *
   * Tree traversal that collects clipped boundary segments.  At each
   * split node:
   *   1. Project the split's projector/threshold to 2D.
   *   2. Project all accumulated ancestor constraints to 2D.
   *   3. Clip the boundary line to the bbox + constraints.
   *   4. Push/pop the current split as a constraint for child traversal.
   */

  BoundaryVisitor::BoundaryVisitor(
      int var_i,
      int var_j,
      std::vector<std::pair<int, types::Feature>> const& fixed_vars,
      types::Feature x_min,
      types::Feature x_max,
      types::Feature y_min,
      types::Feature y_max
  )
      : var_i(var_i)
      , var_j(var_j)
      , fixed_vars(fixed_vars)
      , x_min(x_min)
      , x_max(x_max)
      , y_min(y_min)
      , y_max(y_max)
      , depth(0) {}

  void BoundaryVisitor::visit(TreeCondition const& node) {
    // Project current split to 2D.
    types::FeatureVector proj2d = project_2d(node.projector, var_i, var_j);
    types::Feature thr2d        = adjust_threshold(node.projector, node.threshold, fixed_vars);

    // Project all ancestor constraints to 2D for clipping.
    std::vector<HalfSpace> constraints2d;
    constraints2d.reserve(constraints.size());

    for (auto const& con : constraints) {
      HalfSpace hs2d;
      hs2d.projector = project_2d(con.projector, var_i, var_j);
      hs2d.threshold = adjust_threshold(con.projector, con.threshold, fixed_vars);
      hs2d.is_lower  = con.is_lower;
      constraints2d.push_back(hs2d);
    }

    // Clip the boundary and store if visible.
    BoundarySegment seg;

    if (clip_boundary_2d(proj2d, thr2d, constraints2d, x_min, x_max, y_min, y_max, seg, depth)) {
      segments.push_back(seg);
    }

    // Push this split as a constraint and recurse.
    // Lower child: projector^T x < threshold.
    HalfSpace lower_con;
    lower_con.projector = node.projector;
    lower_con.threshold = node.threshold;
    lower_con.is_lower  = true;
    constraints.push_back(lower_con);

    depth++;
    node.lower->accept(*this);
    depth--;

    // Upper child: projector^T x >= threshold.
    constraints.back().is_lower = false;

    depth++;
    node.upper->accept(*this);
    depth--;

    constraints.pop_back();
  }

  void BoundaryVisitor::visit(TreeResponse const&) {
    // Leaf nodes have no boundary to emit.
  }

  /**
   * RegionVisitor
   *
   * At each leaf, computes the convex polygon for that leaf's decision
   * region.  The algorithm:
   *   1. Start with the bounding box as a rectangle (CCW winding).
   *   2. For each ancestor constraint on the root-to-leaf path, project
   *      the constraint to 2D (project_2d + adjust_threshold) and clip
   *      the polygon using Sutherland–Hodgman (clip_polygon_halfspace).
   *   3. If the polygon is non-empty, store it with the leaf's group.
   *
   * The bounding box should match the visible coordinate range in the
   * plot (the padded data range).  When combined with zero scale
   * expansion in ggplot2, this ensures the region polygons exactly
   * tile the visible area with no whitespace gaps.
   *
   * Constraints are accumulated on a stack during tree traversal
   * (push on entry, pop on exit), so each leaf sees exactly its
   * ancestor constraints.
   */

  RegionVisitor::RegionVisitor(
      int var_i,
      int var_j,
      std::vector<std::pair<int, types::Feature>> const& fixed_vars,
      types::Feature x_min,
      types::Feature x_max,
      types::Feature y_min,
      types::Feature y_max
  )
      : var_i(var_i)
      , var_j(var_j)
      , fixed_vars(fixed_vars)
      , x_min(x_min)
      , x_max(x_max)
      , y_min(y_min)
      , y_max(y_max) {}

  void RegionVisitor::visit(TreeCondition const& node) {
    // Push this split as lower constraint, recurse left.
    HalfSpace con;
    con.projector = node.projector;
    con.threshold = node.threshold;
    con.is_lower  = true;
    constraints.push_back(con);

    node.lower->accept(*this);

    // Flip to upper constraint, recurse right.
    constraints.back().is_lower = false;

    node.upper->accept(*this);

    constraints.pop_back();
  }

  void RegionVisitor::visit(TreeResponse const& node) {
    // Start with the bounding box as a rectangle (CCW winding).
    Polygon poly = {{x_min, y_min}, {x_max, y_min}, {x_max, y_max}, {x_min, y_max}};

    // Clip against each ancestor constraint in 2D.
    for (auto const& con : constraints) {
      types::FeatureVector proj2d = project_2d(con.projector, var_i, var_j);
      types::Feature thr2d        = adjust_threshold(con.projector, con.threshold, fixed_vars);

      poly = clip_polygon_halfspace(poly, proj2d, thr2d, con.is_lower);

      if (poly.empty())
        return; // Region unreachable in this 2D slice.
    }

    RegionPolygon region;
    region.vertices        = std::move(poly);
    region.predicted_group = node.value;
    regions.push_back(std::move(region));
  }

  /**
   * Tree Layout
   *
   * Computes (x, y) positions for rendering the tree as a diagram.
   * Uses a recursive bottom-up algorithm:
   *   1. Layout each subtree, tracking its width.
   *   2. Place sibling subtrees side-by-side with a gap.
   *   3. Position the parent above its left child (left-aligned style).
   *   4. y decreases with depth so the root is at the top.
   */

  namespace {
    /// Intermediate result for subtree layout computation.
    struct LayoutSubtree {
      types::Feature center_x; ///< x-coordinate of this subtree's root node.
      types::Feature width;    ///< Total horizontal span of this subtree.
    };

    /// Format a threshold value for edge labels (e.g. "1.50").
    std::string format_threshold(types::Feature t) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << t;
      return oss.str();
    }

    /**
   * @brief Visitor that recursively lays out tree nodes.
   *
   * Carries shared mutable state (depth, node_idx, x_offset) and
   * writes positioned nodes/edges into output vectors.  After
   * accepting a node, the result is available in @c result.
   */
    struct LayoutVisitor : TreeNode::Visitor {
      int depth;
      int& node_idx;
      types::Feature x_offset;
      LayoutParams const& params;
      std::vector<LayoutNode>& nodes;
      std::vector<LayoutEdge>& edges;

      LayoutSubtree result;

      LayoutVisitor(
          int depth,
          int& node_idx,
          types::Feature x_offset,
          LayoutParams const& params,
          std::vector<LayoutNode>& nodes,
          std::vector<LayoutEdge>& edges
      )
          : depth(depth)
          , node_idx(node_idx)
          , x_offset(x_offset)
          , params(params)
          , nodes(nodes)
          , edges(edges)
          , result{0, 0} {}

      void visit(TreeResponse const& /* response */) override {
        types::Feature y_pos = -static_cast<types::Feature>(depth) * params.y_spacing;
        int my_idx           = node_idx;
        node_idx++;

        types::Feature cx = x_offset + params.leaf_w / 2;
        nodes.push_back({cx, y_pos, true, my_idx});

        result = {cx, params.leaf_w + 0.1f};
      }

      void visit(TreeCondition const& condition) override {
        types::Feature y_pos = -static_cast<types::Feature>(depth) * params.y_spacing;
        int my_idx           = node_idx;
        node_idx++;

        // Reserve slot (pre-order), x filled below.
        std::size_t my_slot = nodes.size();
        nodes.push_back({0, y_pos, false, my_idx});

        LayoutVisitor left_visitor(depth + 1, node_idx, x_offset, params, nodes, edges);
        condition.lower->accept(left_visitor);

        LayoutVisitor right_visitor(
            depth + 1, node_idx, x_offset + left_visitor.result.width + params.gap, params, nodes, edges
        );
        condition.upper->accept(right_visitor);

        // Left-aligned: parent centered above left child.
        types::Feature center_x = left_visitor.result.center_x;
        nodes[my_slot].x        = center_x;

        // Compute edge endpoints (parent bottom -> child top).
        types::Feature from_y  = y_pos - params.node_h / 2;
        types::Feature child_y = -static_cast<types::Feature>(depth + 1) * params.y_spacing;

        types::Feature left_child_h  = condition.lower->is_leaf() ? params.leaf_h : params.node_h;
        types::Feature right_child_h = condition.upper->is_leaf() ? params.leaf_h : params.node_h;

        std::string thr = format_threshold(condition.threshold);

        edges.push_back({center_x, from_y, left_visitor.result.center_x, child_y + left_child_h / 2, "< " + thr});

        edges.push_back({
            center_x,
            from_y,
            right_visitor.result.center_x,
            child_y + right_child_h / 2,
            "\xe2\x89\xa5 " + thr // UTF-8 for ≥
        });

        types::Feature total_width = left_visitor.result.width + params.gap + right_visitor.result.width;
        types::Feature min_width   = params.node_w + 0.1f;

        result = {center_x, std::max(total_width, min_width)};
      }
    };
  }

  TreeLayout compute_tree_layout(TreeNode const& root, LayoutParams const& params) {
    std::vector<LayoutNode> nodes;
    std::vector<LayoutEdge> edges;
    int node_idx = 0;

    LayoutVisitor visitor(0, node_idx, 0.0f, params, nodes, edges);
    root.accept(visitor);

    return {std::move(nodes), std::move(edges)};
  }
}
