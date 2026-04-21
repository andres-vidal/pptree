#include "models/Tree.hpp"

#include "models/ClassificationTree.hpp"
#include "models/RegressionTree.hpp"
#include "models/TreeBranch.hpp"
#include "models/TreeLeaf.hpp"
#include "models/VIVisitor.hpp"
#include "models/strategies/NodeContext.hpp"
#include "stats/Stats.hpp"

#include <stack>
#include <Eigen/Dense>

using namespace ppforest2::pp;
using namespace ppforest2::stats;
using namespace ppforest2::types;

namespace ppforest2 {
  // ---------------------------------------------------------------------------
  // Shared tree-construction algorithm (static Tree::build_root).
  // ---------------------------------------------------------------------------
  namespace {
    TreeNode::Ptr degenerate_leaf(TrainingSpec const& spec, NodeContext const& ctx, stats::RNG& rng) {
      auto leaf        = spec.create_leaf(ctx, rng);
      leaf->degenerate = true;
      return leaf;
    }

    void orient_branches(
        GroupId& group_a,
        GroupId& group_b,
        FeatureMatrix const& x,
        GroupPartition const& y,
        Projector const& projector
    ) {
      Feature mean_a = y.group(x, group_a).colwise().mean().dot(projector);
      Feature mean_b = y.group(x, group_b).colwise().mean().dot(projector);

      if (mean_a > mean_b) {
        std::swap(group_a, group_b);
      }
    }

    struct Step {
      GroupPartition y;
      TreeNode::Ptr* node;
      int depth;

      bool pop               = false;
      TreeNode::Ptr upper    = nullptr;
      TreeNode::Ptr lower    = nullptr;
      Feature cutpoint       = 0;
      Feature pp_index_value = 0;
      Projector projector;

      Step(GroupPartition const& y, TreeNode::Ptr* node, int const cols, int depth = 0)
          : y(y)
          , node(node)
          , depth(depth)
          , projector(Projector::Zero(cols)) {}
    };

    void push_children(
        Step& step,
        NodeContext const& ctx,
        GroupPartition const& lower_y,
        GroupPartition const& upper_y,
        FeatureMatrix const& x,
        std::stack<Step>& stack
    ) {
      step.projector      = ctx.projector;
      step.cutpoint       = ctx.cutpoint;
      step.pp_index_value = ctx.pp_index_value;

      stack.emplace(lower_y, &step.lower, x.cols(), step.depth + 1);
      stack.emplace(upper_y, &step.upper, x.cols(), step.depth + 1);

      step.pop = true;
    }
  }

  TreeNode::Ptr Tree::build_root(
      TrainingSpec const& spec,
      FeatureMatrix const& x,
      GroupPartition const& y,
      stats::RNG& rng,
      FeatureMatrix* mutable_x,
      OutcomeVector* mutable_y
  ) {
    std::stack<Step> stack;

    TreeNode::Ptr root;

    stack.emplace(y, &root, x.cols());

    while (!stack.empty()) {
      Step& step = stack.top();

      if (step.pop) {
        *step.node = TreeBranch::make(
            step.projector,
            step.cutpoint,
            std::move(step.lower),
            std::move(step.upper),
            step.y.groups,
            step.pp_index_value
        );

        stack.pop();
        continue;
      }

      FeatureMatrix const& active_x = mutable_x ? *mutable_x : x;
      NodeContext ctx(active_x, step.y, step.depth);

      // For regression, thread the raw y vector and mutable pointers through.
      ctx.y_vec         = mutable_y;
      ctx.mutable_x     = mutable_x;
      ctx.mutable_y_vec = mutable_y;

      // 1. Stop rule.
      if (spec.should_stop(ctx, rng)) {
        *step.node = spec.create_leaf(ctx, rng);
        stack.pop();
        continue;
      }

      // Single-group check (regression edge case): after ByCutpoint re-clustering,
      // a child may end up with only one group. Create a leaf since neither the
      // binary nor multiclass path can handle it.
      if (step.y.groups.size() < 2) {
        *step.node = spec.create_leaf(ctx, rng);
        stack.pop();
        continue;
      }

      // 2. Variable selection.
      spec.select_vars(ctx, rng);

      // Binary case (2 groups): no binarization, direct projector + cutpoint + grouping.
      if (step.y.groups.size() == 2) {
        GroupId group_1 = *step.y.groups.begin();
        GroupId group_2 = *std::next(step.y.groups.begin());

        spec.find_projection(ctx, rng);

        if (ctx.projector.hasNaN()) {
          *step.node = degenerate_leaf(spec, ctx, rng);
          stack.pop();
          continue;
        }

        spec.find_cutpoint(ctx, rng);

        orient_branches(group_1, group_2, active_x, step.y, ctx.projector);

        ctx.binary_y.emplace(step.y);
        ctx.binary_0 = group_1;
        ctx.binary_1 = group_2;

        auto [lower_y, upper_y] = spec.group(ctx, rng);

        // No-progress guard: if either child covers the same row count as the
        // parent, the grouping strategy failed to split (e.g. ByCutpoint with
        // all rows on one side of the cutpoint). Recursing on the identical
        // partition would produce unbounded recursion — force a leaf instead.
        int const parent_size = step.y.total_size();
        if (lower_y.total_size() >= parent_size || upper_y.total_size() >= parent_size) {
          *step.node = degenerate_leaf(spec, ctx, rng);
          stack.pop();
          continue;
        }

        push_children(step, ctx, lower_y, upper_y, active_x, stack);
        continue;
      }

      // Multiclass case (>2 groups): two projections with binarization in between.
      spec.find_projection(ctx, rng);

      if (ctx.projector.hasNaN()) {
        *step.node = degenerate_leaf(spec, ctx, rng);
        stack.pop();
        continue;
      }

      spec.regroup(ctx, rng);

      if (!ctx.binary_y.has_value() || ctx.binary_y->groups.size() < 2) {
        *step.node = degenerate_leaf(spec, ctx, rng);
        stack.pop();
        continue;
      }

      spec.find_projection(ctx, rng);

      if (ctx.projector.hasNaN()) {
        *step.node = degenerate_leaf(spec, ctx, rng);
        stack.pop();
        continue;
      }

      spec.find_cutpoint(ctx, rng);

      orient_branches(ctx.binary_0, ctx.binary_1, active_x, *ctx.binary_y, ctx.projector);

      auto [lower_y, upper_y] = spec.group(ctx, rng);

      // No-progress guard (see binary-path comment above).
      {
        int const parent_size = step.y.total_size();
        if (lower_y.total_size() >= parent_size || upper_y.total_size() >= parent_size) {
          *step.node = degenerate_leaf(spec, ctx, rng);
          stack.pop();
          continue;
        }
      }

      push_children(step, ctx, lower_y, upper_y, active_x, stack);
    }

    return root;
  }

  // ---------------------------------------------------------------------------
  // Variable importance (tree-level)
  // ---------------------------------------------------------------------------

  FeatureVector Tree::vi_projections(int n_vars, FeatureVector const* scale) const {
    FeatureVector importance = FeatureVector::Zero(n_vars);

    VIVisitor visitor(n_vars, scale);
    root->accept(visitor);

    for (int j = 0; j < n_vars; ++j) {
      importance(j) = static_cast<Feature>(visitor.vi2_contributions[static_cast<std::size_t>(j)]);
    }

    return importance;
  }

  VariableImportance Tree::variable_importance(FeatureMatrix const& x) const {
    VariableImportance vi;
    vi.scale       = stats::sd(x);
    vi.scale       = (vi.scale.array() > Feature(0)).select(vi.scale, Feature(1));
    vi.projections = vi_projections(static_cast<int>(x.cols()), &vi.scale);
    return vi;
  }

  // ---------------------------------------------------------------------------
  // Instance methods
  // ---------------------------------------------------------------------------


  Tree::Tree(TreeNode::Ptr root, TrainingSpec::Ptr training_spec)
      : root(std::move(root)) {
    this->training_spec = std::move(training_spec);
    degenerate          = this->root && this->root->degenerate;
  }

  Outcome Tree::predict(FeatureVector const& data) const {
    return root->predict(data);
  }

  OutcomeVector Tree::predict(FeatureMatrix const& data) const {
    OutcomeVector predictions(data.rows());

    for (int i = 0; i < data.rows(); ++i) {
      predictions(i) = predict(static_cast<FeatureVector>(data.row(i)));
    }

    return predictions;
  }

  bool Tree::operator==(Tree const& other) const {
    return *root == *other.root;
  }

  bool Tree::operator!=(Tree const& other) const {
    return !(*this == other);
  }

  // ---------------------------------------------------------------------------
  // Static factories — dispatch on training_spec.mode
  // ---------------------------------------------------------------------------

  Tree::Ptr Tree::train(TrainingSpec const& training_spec, FeatureMatrix& x, OutcomeVector& y) {
    stats::RNG rng(training_spec.seed);
    return Tree::train(training_spec, x, y, rng);
  }

  Tree::Ptr Tree::train(
      TrainingSpec const& training_spec,
      FeatureMatrix& x,
      GroupPartition const& group_spec,
      OutcomeVector* y_vec
  ) {
    stats::RNG rng(training_spec.seed);
    return Tree::train(training_spec, x, group_spec, rng, y_vec);
  }

  Tree::Ptr Tree::train(
      TrainingSpec const& training_spec,
      FeatureMatrix& x,
      OutcomeVector& y,
      stats::RNG& rng
  ) {
    GroupPartition const group_spec = training_spec.init_groups(y);
    return Tree::train(training_spec, x, group_spec, rng, &y);
  }

  Tree::Ptr Tree::train(
      TrainingSpec const& training_spec,
      FeatureMatrix& x,
      GroupPartition const& group_spec,
      stats::RNG& rng,
      OutcomeVector* y_vec
  ) {
    if (training_spec.mode == types::Mode::Regression) {
      invariant(y_vec != nullptr, "Regression Tree::train requires a response vector");
      // No copy: `x` and `*y_vec` are already mutable buffers owned by the
      // caller (R bindings and CLI discard them after training).
      return RegressionTree::train(training_spec, x, group_spec, rng, *y_vec);
    }

    // Classification doesn't mutate; the const-ref binding is a no-op conversion.
    return ClassificationTree::train(training_spec, x, group_spec, rng);
  }
}
