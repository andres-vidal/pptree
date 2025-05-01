#pragma once

#include "TreeNode.hpp"

namespace models {
  template<typename T, typename R>
  struct TreeCondition : public TreeNode<T, R> {
    pp::Projector<T> projector;
    T threshold;
    std::unique_ptr<TreeNode<T, R> > lower;
    std::unique_ptr<TreeNode<T, R> > upper;
    std::unique_ptr<TrainingSpec<T, R> > training_spec;

    const stats::SortedDataSpec<T, R> training_data;

    TreeCondition(
      const pp::Projector<T>&              projector,
      const Threshold<T>&                  threshold,
      std::unique_ptr<TreeNode<T, R> >     lower,
      std::unique_ptr<TreeNode<T, R> >     upper,
      std::unique_ptr<TrainingSpec<T, R> > training_spec,
      const stats::SortedDataSpec<T, R> &  training_data)
      : projector(projector),
      threshold(threshold),
      lower(std::move(lower)),
      upper(std::move(upper)),
      training_spec(std::move(training_spec)),
      training_data(training_data) {
    }

    TreeCondition(
      const pp::Projector<T>&          projector,
      const Threshold<T>&              threshold,
      std::unique_ptr<TreeNode<T, R> > lower,
      std::unique_ptr<TreeNode<T, R> > upper)
      : projector(projector),
      threshold(threshold),
      lower(std::move(lower)),
      upper(std::move(upper)) {
    }

    void accept(TreeNodeVisitor<T, R> &visitor) const override {
      visitor.visit(*this);
    }

    R response() const override {
      throw std::invalid_argument("Cannot get response from a condition node");
    }

    R predict(const stats::DataColumn<T> &data) const override {
      T projected_data = data.dot(projector);

      if (projected_data < threshold) {
        return lower->predict(data);
      } else {
        return upper->predict(data);
      }
    }

    bool equals(const TreeNode<T, R> &other) const override {
      const auto *cond = dynamic_cast<const TreeCondition<T, R> *>(&other);

      return cond
             && math::collinear(projector, cond->projector)
             && math::is_approx(threshold, cond->threshold)
             && *lower == *(cond->lower)
             && *upper == *(cond->upper);
    }

    json to_json() const override {
      return json{
        { "projector", projector },
        { "threshold", threshold },
        { "lower", lower->to_json() },
        { "upper", upper->to_json() }
      };
    }
  };

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const TreeCondition<T, R>& condition) {
    return ostream << condition.to_json().dump(2, ' ', false);
  }
}
