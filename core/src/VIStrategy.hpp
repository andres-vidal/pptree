#pragma once


#include "Tree.hpp"


#include "BootstrapTree.hpp"
#include "Forest.hpp"
#include "Invariant.hpp"

namespace models {
  template <typename T, typename R>
  class VIStrategy;

  template <typename T, typename R>
  class NodeSummarizer : public TreeNodeVisitor<T, R> {
    private:
      const VIStrategy<T, R>& strategy;
      const stats::Data<T> &       training_x;
      const stats::DataColumn<R> & training_y;

    public:
      math::DVector<T> importance;
      std::set<int> classes;

      explicit NodeSummarizer(
        const VIStrategy<T, R> &     strategy,
        const stats::Data<T> &       training_x,
        const stats::DataColumn<R> & training_y)  :
        strategy(strategy),
        training_x(training_x),
        training_y(training_y),
        importance(math::DVector<T>::Zero(training_x.cols())),
        classes({}) {
      }

      void visit(const TreeCondition<T, R> &condition) override {
        NodeSummarizer lower_summarizer(strategy, training_x, training_y);
        NodeSummarizer upper_summarizer(strategy, training_x, training_y);

        condition.lower->accept(lower_summarizer);
        condition.upper->accept(upper_summarizer);

        classes.insert(lower_summarizer.classes.begin(), lower_summarizer.classes.end());
        classes.insert(upper_summarizer.classes.begin(), upper_summarizer.classes.end());

        importance = strategy.compute_partial(
          lower_summarizer.importance,
          upper_summarizer.importance,
          condition,
          *this,
          training_x,
          training_y);
      }

      void visit(const TreeResponse<T, R> &response) override {
        classes = { response.value };
      }
  };

  template <typename T, typename R>
  struct VIStrategy {
    virtual math::DVector<T> operator()(const Tree<T, R> &tree) const          = 0;
    virtual math::DVector<T> operator()(const BootstrapTree<T, R> &tree) const = 0;
    virtual math::DVector<T> operator()(const Forest<T, R> &forest) const      = 0;

    friend class NodeSummarizer<T, R>;

    private:

      virtual math::DVector<T> compute_partial(
        const math::DVector<T> &     lower_importance,
        const math::DVector<T> &     upper_importance,
        const TreeCondition<T, R> &  condition,
        const NodeSummarizer<T, R> & condition_summary,
        const stats::Data<T> &       training_x,
        const stats::DataColumn<R> & training_y) const = 0;

      virtual math::DVector<T> compute_final(
        const math::DVector<T> &    accumulated_importance,
        const Tree<T, R> &          tree,
        const NodeSummarizer<T, R> &root_summary) const = 0;

      virtual math::DVector<T> compute_final(
        const math::DVector<T> &    accumulated_importance,
        const BootstrapTree<T, R> & tree,
        const NodeSummarizer<T, R> &root_summary) const = 0;

      virtual math::DVector<T> compute_final(
        const math::DVector<T> &accumulated_importance,
        const Forest<T, R> &    forest) const = 0;
  };

  template <typename T, typename R>
  struct BaseVIStrategy : public VIStrategy<T, R> {
    virtual math::DVector<T> operator()(const Tree<T, R> &tree) const override {
      stats::Data<T> x       = models::stats::standardize(tree.x);
      stats::DataColumn<R> y = tree.y;

      Tree<T, R> std_tree = tree.retrain(x, y);

      NodeSummarizer<T, R> summarizer(
        *this,
        std_tree.x,
        std_tree.y);

      std_tree.root->accept(summarizer);

      return compute_final(summarizer.importance, std_tree, summarizer);
    }

    virtual math::DVector<T> operator()(const BootstrapTree<T, R> &tree) const override {
      stats::Data<T> x       = models::stats::standardize(tree.x);
      stats::DataColumn<R> y = tree.y;

      BootstrapTreePtr<T, R> std_tree = tree.retrain(x, y, tree.iob_indices);

      stats::Data<T> sampled_x       = std_tree->x(std_tree->iob_indices, Eigen::all);
      stats::DataColumn<R> sampled_y = std_tree->y(std_tree->iob_indices, Eigen::all);

      NodeSummarizer<T, R> summarizer(
        *this,
        sampled_x,
        sampled_y);

      std_tree->root->accept(summarizer);

      return compute_final(summarizer.importance, *std_tree, summarizer);
    }

    virtual math::DVector<T> operator()(const Forest<T, R> &forest) const override {
      stats::Data<T> x       = models::stats::standardize(forest.x);
      stats::DataColumn<R> y = forest.y;

      Forest<T, R> std_forest = forest.retrain(x, y);

      math::DVector<T> accumulated_importance = std::accumulate(
        std_forest.trees.begin(),
        std_forest.trees.end(),
        math::DVector<T>(math::DVector<T>::Zero(std_forest.x.cols())),
        [this] (math::DVector<T> acc, const std::unique_ptr<BootstrapTree<T, R> >& tree) -> math::DVector<T> {
          return acc + this->operator()(*tree);
        });

      return compute_final(accumulated_importance, std_forest);
    }

    virtual math::DVector<T> compute_final(
      const math::DVector<T> &    accumulated_importance,
      const Tree<T, R> &          tree,
      const NodeSummarizer<T, R> &root_summary) const override {
      return accumulated_importance;
    }

    virtual math::DVector<T> compute_final(
      const math::DVector<T> &    accumulated_importance,
      const BootstrapTree<T, R> & tree,
      const NodeSummarizer<T, R> &root_summary) const override {
      return accumulated_importance;
    }

    virtual math::DVector<T> compute_final(
      const math::DVector<T> &accumulated_importance,
      const Forest<T, R> &    forest) const override {
      return accumulated_importance / forest.trees.size();
    }
  };

  template <typename T, typename R>
  struct VIProjectorStrategy : public BaseVIStrategy<T, R> {
    virtual math::DVector<T> compute_partial(
      const math::DVector<T> &     lower_importance,
      const math::DVector<T> &     upper_importance,
      const TreeCondition<T, R> &  condition,
      const NodeSummarizer<T, R> & condition_summary,
      const stats::Data<T> &       training_x,
      const stats::DataColumn<R> & training_y) const override {
      const int n_classes = condition_summary.classes.size();

      return (condition.projector.array().abs() / n_classes).matrix() + lower_importance + upper_importance;
    }
  };

  template <typename T, typename R>
  struct VIProjectorAdjustedStrategy : public BaseVIStrategy<T, R> {
    using Base = BaseVIStrategy<T, R>;
    using Base::operator();

    virtual math::DVector<T> operator()(const Tree<T, R> &tree) const override {
      throw std::invalid_argument("VIProjectorAdjustedStrategy not supported for Tree");
    }

    virtual math::DVector<T> compute_partial(
      const math::DVector<T> &     lower_importance,
      const math::DVector<T> &     upper_importance,
      const TreeCondition<T, R> &  condition,
      const NodeSummarizer<T, R> & condition_summary,
      const stats::Data<T> &       training_x,
      const stats::DataColumn<R> & training_y) const override {
      invariant(condition.training_spec != nullptr, "training_spec is null");
      invariant(condition.training_spec->pp_strategy != nullptr, "pp_strategy is null");

      stats::GroupSpec<R> data_spec(training_y);

      const float pp_index = condition.training_spec->pp_strategy->index(
        training_x,
        data_spec.subset(condition.classes),
        condition.projector);

      return (condition.projector.array().abs() * pp_index).matrix() + lower_importance + upper_importance;
    }

    virtual math::DVector<T> compute_final(
      const math::DVector<T> &     accumulated_importance,
      const BootstrapTree<T, R> &  tree,
      const NodeSummarizer<T, R> & root_summary) const override {
      const int n_classes   = root_summary.classes.size();
      const float oob_error = tree.error_rate();
      return ((1 - oob_error) / (n_classes - 1)) * accumulated_importance;
    }
  };

  template <typename T, typename R>
  struct VIPermutationStrategy : public BaseVIStrategy<T, R> {
    using Base = BaseVIStrategy<T, R>;
    using Base::operator();


    virtual math::DVector<T> operator()(const Tree<T, R> &tree) const override {
      throw std::invalid_argument("VIPermutationStrategy not supported for Tree");
    }

    virtual math::DVector<T> compute_partial(
      const math::DVector<T> &     lower_importance,
      const math::DVector<T> &     upper_importance,
      const TreeCondition<T, R> &  condition,
      const NodeSummarizer<T, R> & condition_summary,
      const stats::Data<T> &       training_x,
      const stats::DataColumn<R> & training_y) const override {
      return math::DVector<T>::Zero(lower_importance.size());
    }

    virtual math::DVector<T> compute_final(
      const math::DVector<T> &     accumulated_importance,
      const BootstrapTree<T, R> &  tree,
      const NodeSummarizer<T, R> & root_summary) const override {
      std::vector<int> oob_indices_vec(tree.oob_indices.begin(), tree.oob_indices.end());

      const stats::Data<T> oob_x       = tree.x(oob_indices_vec, Eigen::all);
      const stats::DataColumn<R> oob_y = tree.y(oob_indices_vec, Eigen::all);

      const stats::DataColumn<R> oob_predictions = tree.predict(oob_x);

      const float oob_accuracy = stats::accuracy(oob_predictions, oob_y);

      math::DVector<T> importance = math::DVector<T>(oob_x.cols());

      for (int j = 0; j < oob_x.cols(); j++) {
        const stats::Data<T> nonsense_data              = stats::shuffle_column(oob_x, j);
        const stats::DataColumn<R> nonsense_predictions = tree.predict(nonsense_data);
        const float nonsense_accuracy                   = stats::accuracy(nonsense_predictions, oob_y);

        importance(j) = oob_accuracy - nonsense_accuracy;
      }

      return importance;
    };
  };
};
