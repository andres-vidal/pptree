#pragma once

#include "TreeNode.hpp"

namespace models {
  template<typename T, typename R>
  struct Condition : public Node<T, R> {
    pp::Projector<T> projector;
    T threshold;
    std::unique_ptr<Node<T, R> > lower;
    std::unique_ptr<Node<T, R> > upper;
    std::unique_ptr<TrainingSpec<T, R> > training_spec;

    const stats::SortedDataSpec<T, R> training_data;

    Condition(
      const pp::Projector<T>&              projector,
      const Threshold<T>&                  threshold,
      std::unique_ptr<Node<T, R> >         lower,
      std::unique_ptr<Node<T, R> >         upper,
      std::unique_ptr<TrainingSpec<T, R> > training_spec,
      const stats::SortedDataSpec<T, R> &  training_data)
      : projector(projector),
      threshold(threshold),
      lower(std::move(lower)),
      upper(std::move(upper)),
      training_spec(std::move(training_spec)),
      training_data(training_data) {
    }

    Condition(
      const pp::Projector<T>&      projector,
      const Threshold<T>&          threshold,
      std::unique_ptr<Node<T, R> > lower,
      std::unique_ptr<Node<T, R> > upper)
      : projector(projector),
      threshold(threshold),
      lower(std::move(lower)),
      upper(std::move(upper)) {
    }

    void accept(NodeVisitor<T, R> &visitor) const override {
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

    bool operator==(const Condition<T, R> &other) const {
      return math::collinear(projector, other.projector)
             && math::is_approx(threshold, other.threshold)
             && *lower == *other.lower
             && *upper == *other.upper;
    }

    bool operator!=(const Condition<T, R> &other) const {
      return !(*this == other);
    }

    json to_json() const override {
      return json{
        { "projector", projector },
        { "threshold", threshold },
        { "lower", lower->to_json() },
        { "upper", upper->to_json() }
      };
    }

    bool equals(const Node<T, R> &other) const override {
      return other.equals(*this);
    }

    bool equals(const Condition<T, R> &other) const override {
      return *this == other;
    }

    bool equals(const Response<T, R> &other) const override {
      return false;
    }
  };
}
