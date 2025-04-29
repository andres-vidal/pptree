#pragma once

#include "Node.hpp"
#include "Tree.hpp"
#include "BootstrapTree.hpp"
#include "SortedDataSpec.hpp"
#include "Forest.hpp"
#include "Invariant.hpp"

namespace models {
  template <typename T, typename R>
  class VIStrategy;

  template <typename T, typename R>
  class NodeSummarizer : public NodeVisitor<T, R> {
    private:
      const VIStrategy<T, R>& strategy;

    public:
      math::DVector<T> importance;
      std::set<int> classes;

      explicit NodeSummarizer(
        const VIStrategy<T, R> &strategy,
        const int               n_vars)  :
        strategy(strategy),
        importance(math::DVector<T>::Zero(n_vars)),
        classes({}) {
      }

      void visit(const Condition<T, R> &condition) override {
        NodeSummarizer lower_summarizer(strategy, importance.size());
        NodeSummarizer upper_summarizer(strategy, importance.size());

        condition.lower->accept(lower_summarizer);
        condition.upper->accept(upper_summarizer);

        classes.insert(lower_summarizer.classes.begin(), lower_summarizer.classes.end());
        classes.insert(upper_summarizer.classes.begin(), upper_summarizer.classes.end());

        importance = strategy.compute_partial(
          lower_summarizer.importance,
          upper_summarizer.importance,
          condition,
          *this);
      }

      void visit(const Response<T, R> &response) override {
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
        const Condition<T, R> &      condition,
        const NodeSummarizer<T, R> & condition_summary) const = 0;

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
      Tree<T, R> std_tree = tree.retrain(tree.training_data->standardize());

      NodeSummarizer<T, R> summarizer(*this, std_tree.training_data->x.cols());
      std_tree.root->accept(summarizer);

      return compute_final(summarizer.importance, std_tree, summarizer);
    }

    virtual math::DVector<T> operator()(const BootstrapTree<T, R> &tree) const override {
      BootstrapTree<T, R> std_tree = tree.retrain(
        models::stats::BootstrapDataSpec<T, R>(
          tree.training_data->standardize(),
          tree.training_data->sample_indices)
        );

      NodeSummarizer<T, R> summarizer(*this, std_tree.training_data->x.cols());
      std_tree.root->accept(summarizer);

      return compute_final(summarizer.importance, std_tree, summarizer);
    }

    virtual math::DVector<T> operator()(const Forest<T, R> &forest) const override {
      Forest<T, R> std_forest = forest.retrain(forest.training_data->standardize());

      math::DVector<T> accumulated_importance = std::accumulate(
        std_forest.trees.begin(),
        std_forest.trees.end(),
        math::DVector<T>(math::DVector<T>::Zero(std_forest.training_data->x.cols())),
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
      const Condition<T, R> &      condition,
      const NodeSummarizer<T, R> & condition_summary) const override {
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
      const Condition<T, R> &      condition,
      const NodeSummarizer<T, R> & condition_summary) const override {
      invariant(condition.training_data != nullptr, "training_data is null");
      invariant(condition.training_spec != nullptr, "training_spec is null");
      invariant(condition.training_spec->pp_strategy != nullptr, "pp_strategy is null");

      stats::DataSpec<T, R> data = condition.training_data->get();

      const float pp_index = condition.training_spec->pp_strategy->index(
        data,
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
      const Condition<T, R> &      condition,
      const NodeSummarizer<T, R> & condition_summary) const override {
      return math::DVector<T>::Zero(lower_importance.size());
    }

    virtual math::DVector<T> compute_final(
      const math::DVector<T> &     accumulated_importance,
      const BootstrapTree<T, R> &  tree,
      const NodeSummarizer<T, R> & root_summary) const override {
      const stats::SortedDataSpec<T, R> oob      = tree.training_data->get_oob();
      const stats::DataColumn<R> oob_predictions = tree.predict(oob.x);

      const float oob_accuracy = stats::accuracy(oob_predictions, oob.y);

      math::DVector<T> importance = math::DVector<T>(oob.x.cols());

      for (int j = 0; j < oob.x.cols(); j++) {
        const stats::Data<T> nonsense_data              = stats::shuffle_column(oob.x, j);
        const stats::DataColumn<R> nonsense_predictions = tree.predict(nonsense_data);
        const float nonsense_accuracy                   = stats::accuracy(nonsense_predictions, oob.y);

        importance(j) = oob_accuracy - nonsense_accuracy;
      }

      return importance;
    };
  };
};
