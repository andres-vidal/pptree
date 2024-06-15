#pragma once

#include "Node.hpp"
#include "Tree.hpp"
#include "BootstrapTree.hpp"
#include "Forest.hpp"

namespace models {
  template <typename T, typename R>
  class VIStrategy;

  template <typename T, typename R>
  class NodeSummarizer : public NodeVisitor<T, R> {
    private:
      const VIStrategy<T, R>& strategy;

    public:
      math::DVector<T> importance;

      explicit NodeSummarizer(
        const VIStrategy<T, R> &strategy,
        const int               n_vars)  :
        strategy(strategy), importance(math::DVector<T>::Zero(n_vars)) {
      }

      void visit(const Condition<T, R> &condition) override {
        NodeSummarizer lower_summarizer(strategy, importance.size());
        NodeSummarizer upper_summarizer(strategy, importance.size());

        condition.lower->accept(lower_summarizer);
        condition.upper->accept(upper_summarizer);

        importance = strategy.compute_partial(
          lower_summarizer.importance,
          upper_summarizer.importance,
          condition);
      }

      void visit(const Response<T, R> &response) override {
      }
  };

  template <typename T, typename R>
  struct VIStrategy {
    virtual math::DVector<T> operator()(const Tree<T, R> &tree) const = 0;
    virtual math::DVector<T> operator()(const BootstrapTree<T, R> &tree) const = 0;
    virtual math::DVector<T> operator()(const Forest<T, R> &forest) const = 0;

    friend class NodeSummarizer<T, R>;

    private:

      virtual math::DVector<T> compute_partial(
        const math::DVector<T> &lower_importance,
        const math::DVector<T> &upper_importance,
        const Condition<T, R> & condition) const = 0;

      virtual math::DVector<T> compute_final(
        const math::DVector<T> &accumulated_importance,
        const Tree<T, R> &      tree) const = 0;

      virtual math::DVector<T> compute_final(
        const math::DVector<T> &   accumulated_importance,
        const BootstrapTree<T, R> &tree) const = 0;

      virtual math::DVector<T> compute_final(
        const math::DVector<T> &accumulated_importance,
        const Forest<T, R> &    forest) const = 0;
  };

  template <typename T, typename R>
  struct BaseVIStrategy : public VIStrategy<T, R> {
    virtual math::DVector<T> operator()(const Tree<T, R> &tree) const override {
      Tree<T, R> std_tree = tree.standardize();

      NodeSummarizer<T, R> summarizer(*this, std_tree.training_data->x.cols());
      std_tree.root->accept(summarizer);

      return compute_final(summarizer.importance, std_tree);
    }

    virtual math::DVector<T> operator()(const BootstrapTree<T, R> &tree) const override {
      BootstrapTree<T, R> std_tree = tree.standardize();

      NodeSummarizer<T, R> summarizer(*this, std_tree.training_data->x.cols());
      std_tree.root->accept(summarizer);

      return compute_final(summarizer.importance, std_tree);
    }

    virtual math::DVector<T> operator()(const Forest<T, R> &forest) const override {
      Forest<T, R> std_forest = forest.standardize();

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
      const math::DVector<T> &accumulated_importance,
      const Tree<T, R> &      tree) const override {
      return accumulated_importance;
    }

    virtual math::DVector<T> compute_final(
      const math::DVector<T> &   accumulated_importance,
      const BootstrapTree<T, R> &tree) const override {
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
      const math::DVector<T> &lower_importance,
      const math::DVector<T> &upper_importance,
      const Condition<T, R> & condition) const override {
      const int n_classes = condition.classes().size();

      return math::abs(condition.projector.vector) / n_classes + lower_importance + upper_importance;
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
      const math::DVector<T> &lower_importance,
      const math::DVector<T> &upper_importance,
      const Condition<T, R> & condition) const override {
      const long double pp_index = condition.projector.index;

      return math::abs(condition.projector.vector) * pp_index + lower_importance + upper_importance;
    }

    virtual math::DVector<T> compute_final(
      const math::DVector<T> &   accumulated_importance,
      const BootstrapTree<T, R> &tree) const override {
      const int n_classes = tree.root->classes().size();
      const long double oob_error = tree.error_rate();
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
      const math::DVector<T> &lower_importance,
      const math::DVector<T> &upper_importance,
      const Condition<T, R> & condition) const override {
      return math::DVector<T>::Zero(lower_importance.size());
    }

    virtual math::DVector<T> compute_final(
      const math::DVector<T> &   _accumulated_importance,
      const BootstrapTree<T, R> &tree) const override {
      const stats::DataSpec<T, R> oob = tree.training_data->get_oob();
      const stats::DataColumn<R> oob_predictions = tree.predict(oob.x);
      const long double oob_accuracy = stats::accuracy(oob_predictions, oob.y);

      math::DVector<T> importance = math::DVector<T>(oob.x.cols());

      for (int j = 0; j < oob.x.cols(); j++) {
        const stats::Data<T> nonsense_data = stats::shuffle_column(oob.x, j);
        const stats::DataColumn<R> nonsense_predictions = tree.predict(nonsense_data);
        const long double nonsense_accuracy = stats::accuracy(nonsense_predictions, oob.y);

        importance(j) = oob_accuracy - nonsense_accuracy;
      }

      return importance;
    };
  };
};
