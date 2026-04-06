#include "models/Tree.hpp"

#include "models/TreeBranch.hpp"
#include "models/TreeLeaf.hpp"
#include "models/BootstrapTree.hpp"
#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"
#include "utils/Map.hpp"

#include <cmath>
#include <map>
#include <stack>
#include <Eigen/Dense>

using namespace ppforest2::pp;
using namespace ppforest2::stats;
using namespace ppforest2::types;
using namespace ppforest2::utils;

namespace ppforest2 {
  namespace {
    TreeNode::Ptr degenerate_leaf(TrainingSpec const& spec, NodeContext const& ctx, stats::RNG& rng) {
      auto leaf        = spec.create_leaf(ctx, rng);
      leaf->degenerate = true;
      return leaf;
    }

    void orient_branches(
        Outcome& group_a, Outcome& group_b, FeatureMatrix const& x, GroupPartition const& y, Projector const& projector
    ) {
      Feature mean_a = y.group(x, group_a).colwise().mean().dot(projector);
      Feature mean_b = y.group(x, group_b).colwise().mean().dot(projector);

      if (mean_a > mean_b) {
        std::swap(group_a, group_b);
      }
    }

    std::pair<Outcome, Outcome> orient_branches(
        Outcome const& group_a,
        Outcome const& group_b,
        FeatureMatrix const& x,
        GroupPartition const& y,
        Projector const& projector
    ) {
      Outcome lower = group_a;
      Outcome upper = group_b;
      orient_branches(lower, upper, x, y, projector);
      return {lower, upper};
    }
  }

  struct Step {
    GroupPartition y;
    TreeNode::Ptr* node;
    int depth;

    bool pop               = false;
    TreeNode::Ptr upper    = nullptr;
    TreeNode::Ptr lower    = nullptr;
    Cutpoint cutpoint      = 0;
    Feature pp_index_value = 0;
    Projector projector;

    Step(GroupPartition const& y, TreeNode::Ptr* node, int const cols, int depth = 0)
        : y(y)
        , node(node)
        , depth(depth)
        , projector(Projector::Zero(cols)) {}
  };

  TreeNode::Ptr build_root(TrainingSpec const& spec, FeatureMatrix const& x, GroupPartition const& y, stats::RNG& rng) {
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

      NodeContext ctx(x, step.y, step.depth);

      /**
       * 1. Stop rule
       *
       * Check if the node should stop growing.
       * If it should, create a leaf node that may be degenerate.
       * Otherwise, continue to the next step.
       */
      if (spec.should_stop(ctx, rng)) {
        *step.node = spec.create_leaf(ctx, rng);
        stack.pop();
        continue;
      }

      /**
       * 2. Variable selection
       *
       * Select a subset of variables for this split. For single trees this
       * is typically all variables; for forests a random subset introduces
       * diversity across trees.
       */
      spec.select_vars(ctx, rng);

      /**
       * Binary case (2 groups)
       *
       * With only two groups there is no need for binarization or a separate
       * partition step — the projector directly separates the two groups,
       * and each group becomes a leaf.
       */
      if (step.y.groups.size() == 2) {
        Outcome const group_1 = *step.y.groups.begin();
        Outcome const group_2 = *std::next(step.y.groups.begin());

        /**
         * 3. Projection
         *
         * Find the projector that maximizes the projection pursuit index
         * on the two groups. A NaN projector indicates a degenerate case
         * (e.g. singular within-group covariance).
         */
        spec.find_projection(ctx, rng);

        if (ctx.projector.hasNaN()) {
          *step.node = degenerate_leaf(spec, ctx, rng);
          stack.pop();
          continue;
        }

        /**
         * 4. Cutpoint
         *
         * Compute the split cutpoint in the projected 1D space, then
         * assign groups to branches by comparing their projected means.
         */
        spec.find_cutpoint(ctx, rng);

        auto [lower_group, upper_group] = orient_branches(group_1, group_2, x, step.y, ctx.projector);

        TreeLeaf::Ptr lower_response = TreeLeaf::make(lower_group);
        TreeLeaf::Ptr upper_response = TreeLeaf::make(upper_group);

        *step.node = TreeBranch::make(
            ctx.projector,
            ctx.cutpoint,
            std::move(lower_response),
            std::move(upper_response),
            step.y.groups,
            ctx.pp_index_value
        );

        stack.pop();
        continue;
      }

      /**
       * Multiclass case (>2 groups)
       *
       * The projection pursuit index is defined for two groups, so multiclass
       * nodes require two projection steps with a binarization step in between:
       *
       *   3. First projection  — on all G groups, used for binarization.
       *   4. Binarization      — reduce G groups to 2 superclasses.
       *   5. Second projection — on the 2 superclasses, used for the split.
       *   6. Cutpoint          — split value in the second projected space.
       *   7. Partition          — route original groups to child nodes.
       */

      /**
       * 3. First projection (G groups)
       *
       * Project all G group means onto 1D. The resulting ordering is used
       * by the binarization strategy to decide which groups go together.
       */
      spec.find_projection(ctx, rng);

      if (ctx.projector.hasNaN()) {
        *step.node = degenerate_leaf(spec, ctx, rng);
        stack.pop();
        continue;
      }

      /**
       * 4. Binarization
       *
       * Reduce the G-group problem to a binary one by grouping the original
       * classes into two superclasses. The default strategy (LargestGap)
       * sorts groups by their projected mean and splits at the largest gap.
       */
      spec.regroup(ctx, rng);

      if (!ctx.binary_y.has_value() || ctx.binary_y->groups.size() < 2) {
        *step.node = degenerate_leaf(spec, ctx, rng);
        stack.pop();
        continue;
      }

      /**
       * 5. Second projection (2 superclasses)
       *
       * Re-run projection pursuit on the binary partition. This projector
       * is optimized for separating the two superclasses and is the one
       * stored in the tree node.
       */
      spec.find_projection(ctx, rng);

      if (ctx.projector.hasNaN()) {
        *step.node = degenerate_leaf(spec, ctx, rng);
        stack.pop();
        continue;
      }

      /**
       * 6. Cutpoint
       *
       * Compute the split cutpoint in the second projected space, then
       * orient the branches so the lower-mean superclass goes left.
       */
      spec.find_cutpoint(ctx, rng);

      orient_branches(ctx.binary_0, ctx.binary_1, x, *ctx.binary_y, ctx.projector);

      /**
       * 7. Partition
       *
       * Route the original groups to child nodes based on the binary
       * mapping. Each child receives a GroupPartition with the subset
       * of groups assigned to its superclass.
       */
      auto [lower_y, upper_y] = spec.split(ctx, rng);

      step.projector      = ctx.projector;
      step.cutpoint       = ctx.cutpoint;
      step.pp_index_value = ctx.pp_index_value;

      stack.emplace(lower_y, &step.lower, x.cols(), step.depth + 1);
      stack.emplace(upper_y, &step.upper, x.cols(), step.depth + 1);

      step.pop = true;
    }

    return root;
  }

  Tree Tree::train(TrainingSpec const& training_spec, FeatureMatrix const& x, OutcomeVector const& y) {
    stats::RNG rng(training_spec.seed);
    return Tree::train(training_spec, x, y, rng);
  }

  Tree Tree::train(TrainingSpec const& training_spec, FeatureMatrix const& x, GroupPartition const& group_spec) {
    stats::RNG rng(training_spec.seed);
    return Tree::train(training_spec, x, group_spec, rng);
  }

  Tree Tree::train(TrainingSpec const& training_spec, FeatureMatrix const& x, OutcomeVector const& y, stats::RNG& rng) {
    GroupPartition const group_spec(y);

    return Tree::train(training_spec, x, group_spec, rng);
  }

  Tree Tree::train(
      TrainingSpec const& training_spec, FeatureMatrix const& x, GroupPartition const& group_spec, stats::RNG& rng
  ) {
    TreeNode::Ptr root_ptr = build_root(training_spec, x, group_spec, rng);

    Tree tree(std::move(root_ptr), TrainingSpec::make(training_spec));

    return tree;
  }

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

    for (int i = 0; i < data.rows(); i++) {
      predictions(i) = predict((FeatureVector)data.row(i));
    }

    return predictions;
  }

  FeatureMatrix Tree::predict(FeatureMatrix const& data, Proportions) const {
    std::set<Outcome> group_set = root->node_groups();
    std::vector<Outcome> groups(group_set.begin(), group_set.end());
    int const G = static_cast<int>(groups.size());

    std::map<Outcome, int> group_to_col;
    for (int g = 0; g < G; ++g) {
      group_to_col[groups[static_cast<std::size_t>(g)]] = g;
    }

    int const n               = static_cast<int>(data.rows());
    FeatureMatrix proportions = FeatureMatrix::Zero(n, G);

    for (int i = 0; i < n; ++i) {
      Outcome const pred                 = predict(static_cast<FeatureVector>(data.row(i)));
      proportions(i, group_to_col[pred]) = Feature(1);
    }

    return proportions;
  }

  bool Tree::operator==(Tree const& other) const {
    return *root == *other.root;
  }

  bool Tree::operator!=(Tree const& other) const {
    return !(*this == other);
  }

  void Tree::accept(Model::Visitor& visitor) const {
    visitor.visit(*this);
  }
}
