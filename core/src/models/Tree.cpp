#include "models/Tree.hpp"

#include "models/ClassificationTree.hpp"
#include "models/Model.hpp"
#include "models/RegressionTree.hpp"
#include "models/TreeBranch.hpp"
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
        GroupPartition const& lower_y_part,
        GroupPartition const& upper_y_part,
        FeatureMatrix const& x,
        std::stack<Step>& stack
    ) {
      step.projector      = *ctx.projector;
      step.cutpoint       = *ctx.cutpoint;
      step.pp_index_value = *ctx.pp_index_value;

      stack.emplace(lower_y_part, &step.lower, x.cols(), step.depth + 1);
      stack.emplace(upper_y_part, &step.upper, x.cols(), step.depth + 1);

      step.pop = true;
    }

  }

  TreeNode::Ptr Tree::build_root(
      TrainingSpec const& spec, FeatureMatrix& x, OutcomeVector& y, GroupPartition const& partition, stats::RNG& rng
  ) {
    std::stack<Step> stack;

    TreeNode::Ptr root;

    stack.emplace(partition, &root, x.cols());

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

      NodeContext ctx(x, step.y, y, step.depth);

      if (spec.should_stop(ctx, rng)) {
        *step.node = spec.create_leaf(ctx, rng);
        stack.pop();
        continue;
      }

      spec.select_vars(ctx, rng);

      spec.find_projection(ctx, rng);
      if (step.y.groups.size() > 2) {
        spec.regroup(ctx, rng);
        spec.find_projection(ctx, rng);
      }
      spec.find_cutpoint(ctx, rng);
      spec.group(ctx, rng);

      if (ctx.aborted) {
        *step.node = degenerate_leaf(spec, ctx, rng);
        stack.pop();
        continue;
      }

      push_children(step, ctx, *ctx.lower_y_part, *ctx.upper_y_part, x, stack);
    }

    return root;
  }

  // ---------------------------------------------------------------------------
  // Variable importance (tree-level)
  // ---------------------------------------------------------------------------

  FeatureVector Tree::vi_projections(int n_vars, FeatureVector const* scale) const {
    VIVisitor visitor(n_vars, scale);
    root->accept(visitor);

    return Eigen::Map<FeatureVector const>(visitor.vi2_contributions.data(), n_vars);
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


  Tree::Tree(TreeNode::Ptr root, TrainingSpec::Ptr spec)
      : root(std::move(root)) {
    this->training_spec = std::move(spec);
    degenerate          = this->root && this->root->degenerate;
  }

  Outcome Tree::predict(FeatureVector const& data) const {
    return root->predict(data);
  }

  OutcomeVector Tree::predict(FeatureMatrix const& data) const {
    return OutcomeVector::NullaryExpr(data.rows(), [&](Eigen::Index i) {
      return predict(static_cast<FeatureVector>(data.row(i)));
    });
  }

  bool Tree::operator==(Tree const& other) const {
    return *root == *other.root;
  }

  bool Tree::operator!=(Tree const& other) const {
    return !(*this == other);
  }

  // ---------------------------------------------------------------------------
  // Static factories — dispatch on spec.mode
  // ---------------------------------------------------------------------------

  Tree::Ptr Tree::train(TrainingSpec const& spec, FeatureMatrix& x, OutcomeVector& y) {
    stats::RNG rng(spec.seed);
    return Tree::train(spec, x, y, rng);
  }

  Tree::Ptr Tree::train(TrainingSpec const& spec, FeatureMatrix& x, OutcomeVector& y, stats::RNG& rng) {
    Model::check_train_inputs(x, y);

    GroupPartition const y_part = spec.init_groups(y);

    Tree::Ptr tree;

    if (spec.mode == types::Mode::Regression) {
      tree = RegressionTree::train(spec, x, y, y_part, rng);
    } else {
      tree = ClassificationTree::train(spec, x, y, y_part, rng);
    }

    return tree;
  }
}
