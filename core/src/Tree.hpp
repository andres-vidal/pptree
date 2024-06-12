#pragma once

#include "DataSpec.hpp"
#include "TrainingSpec.hpp"
#include "Projector.hpp"
#include "ConfusionMatrix.hpp"

#include <nlohmann/json.hpp>

namespace models {
  using json = nlohmann::json;

  template<typename T>
  using Threshold = T;

  template<typename T, typename R>
  struct Node;
  template<typename T, typename R>
  struct Condition;
  template<typename T, typename R>
  struct Response;

  enum class VariableImportanceKind {
    PROJECTOR,
    PROJECTOR_ADJUSTED,
    PERMUTATION,
  };

  template<typename T, typename R>
  struct NodeVisitor {
    virtual void visit(const Condition<T, R> &condition) = 0;
    virtual void visit(const Response<T, R> &response) = 0;
  };

  template<typename T, typename R>
  struct Node {
    virtual ~Node() = default;
    virtual void accept(NodeVisitor<T, R> &visitor) const = 0;
    virtual R predict(const stats::DataColumn<T> &data) const = 0;
    virtual R response() const = 0;
    virtual std::set<int> classes() const = 0;
    virtual int partition_count() const = 0;
    virtual math::DVector<T> variable_importance(VariableImportanceKind) const = 0;
    virtual json to_json() const = 0;
    virtual bool equals(const Node<T, R> &other) const = 0;
    virtual bool equals(const Condition<T, R> &other) const = 0;
    virtual bool equals(const Response<T, R> &other) const = 0;


    bool operator==(const Node<T, R> &other) const {
      return this->equals(other);
    }

    bool operator!=(const Node<T, R> &other) const {
      return this->equals(other);
    }
  };

  template<typename T, typename R>
  struct Condition : public Node<T, R> {
    pp::Projector<T> projector;
    Threshold<T> threshold;
    std::unique_ptr<Node<T, R> > lower;
    std::unique_ptr<Node<T, R> > upper;

    Condition(
      pp::Projector<T>             projector,
      Threshold<T>                 threshold,
      std::unique_ptr<Node<T, R> > lower,
      std::unique_ptr<Node<T, R> > upper)
      : projector(projector), threshold(threshold), lower(std::move(lower)), upper(std::move(upper)) {
    }

    void accept(NodeVisitor<T, R> &visitor) const override {
      visitor.visit(*this);
    }

    R response() const override {
      throw std::invalid_argument("Cannot get response from a condition node");
    }

    R predict(const stats::DataColumn<T> &data) const override {
      T projected_data = projector.project(data);

      if (projected_data < threshold) {
        return lower->predict(data);
      } else {
        return upper->predict(data);
      }
    }

    bool operator==(const Condition<T, R> &other) const {
      return math::collinear(projector.vector, other.projector.vector)
             && math::is_approx(threshold, other.threshold)
             && *lower == *other.lower
             && *upper == *other.upper;
    }

    bool operator!=(const Condition<T, R> &other) const {
      return !(*this == other);
    }

    std::set<int> classes() const override {
      std::set<int> classes;
      auto [lower_classes, upper_classes] = std::make_tuple(lower->classes(), upper->classes());
      classes.insert(lower_classes.begin(), lower_classes.end());
      classes.insert(upper_classes.begin(), upper_classes.end());
      return classes;
    }

    int partition_count() const override {
      return 1 + lower->partition_count() + upper->partition_count();
    }

    math::DVector<T> variable_importance(VariableImportanceKind importance_kind) const override {
      math::DVector<T> lower_importance = lower->variable_importance(importance_kind);
      math::DVector<T> upper_importance = upper->variable_importance(importance_kind);

      double factor = 1 / (double)classes().size();

      if (importance_kind == VariableImportanceKind::PROJECTOR_ADJUSTED) {
        factor = projector.index;
      }

      math::DVector<T> importance = math::abs(projector.vector) * factor;

      if (lower_importance.size()) {
        importance += lower_importance;
      }

      if (upper_importance.size()) {
        importance += upper_importance;
      }

      return importance;
    }

    json to_json() const override {
      return json{
        { "projector", projector.to_json() },
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

  template<typename T, typename R>
  struct Response : public Node<T, R> {
    R value;

    explicit Response(R value) : value(value) {
    }

    void accept(NodeVisitor<T, R> &visitor) const override {
      visitor.visit(*this);
    }

    R response() const override {
      return value;
    }

    R predict(const stats::DataColumn<T> &data) const override {
      return value;
    }

    bool operator==(const Response<T, R> &other) const {
      return value == other.value;
    }

    bool operator!=(const Response<T, R> &other) const {
      return !(*this == other);
    }

    std::set<int> classes() const override {
      return { value };
    }

    math::DVector<T> variable_importance(VariableImportanceKind) const override {
      return math::DVector<T>::Zero(0);
    }

    json to_json() const override {
      return json{
        { "value", value }
      };
    }

    bool equals(const Node<T, R> &other) const override {
      return other.equals(*this);
    }

    bool equals(const Condition<T, R> &other) const override {
      return false;
    }

    bool equals(const Response<T, R> &other) const override {
      return *this == other;
    }

    int partition_count() const override {
      return 0;
    }
  };

  template<typename T, typename R, typename D = stats::DataSpec<T, R> >
  struct Tree {
    std::unique_ptr<Condition<T, R> > root;
    std::unique_ptr<TrainingSpec<T, R> > training_spec;
    std::shared_ptr<D> training_data;

    explicit Tree(std::unique_ptr<Condition<T, R> > root) : root(std::move(root)) {
    }

    Tree(
      std::unique_ptr<Condition<T, R> >    root,
      std::unique_ptr<TrainingSpec<T, R> > training_spec,
      std::shared_ptr<D >                  training_data)
      : root(std::move(root)),
      training_spec(std::move(training_spec)),
      training_data(training_data) {
    }

    R predict(const stats::DataColumn<T> &data) const {
      return root->predict(data);
    }

    stats::DataColumn<R> predict(const stats::Data<T> &data) const {
      stats::DataColumn<R> predictions(data.rows());

      for (int i = 0; i < data.rows(); i++) {
        predictions(i) = predict((stats::DataColumn<T>)data.row(i));
      }

      return predictions;
    }

    bool operator==(const Tree<T, R, D> &other) const {
      return *root == *other.root;
    }

    bool operator!=(const Tree<T, R, D> &other) const {
      return !(*this == other);
    }

    Tree<T, R, D > retrain(const D &data) const {
      return train(*training_spec, data);
    }

    math::DVector<T> variable_importance() const {
      return variable_importance(VariableImportanceKind::PROJECTOR);
    }

    virtual double error_rate(const stats::DataSpec<T, R> &data) const {
      auto [x, y, _classes] = data.unwrap();
      return stats::error_rate(predict(x), y);
    }

    virtual stats::ConfusionMatrix confusion_matrix(const stats::DataSpec<T, R> &data) const {
      auto [x, y, _classes] = data.unwrap();
      return stats::ConfusionMatrix(predict(x), y);
    }

    json to_json() const {
      return json{
        { "root", root->to_json() }
      };
    }

    protected:
      virtual math::DVector<T> variable_importance(VariableImportanceKind importance_kind) const {
        if (importance_kind == VariableImportanceKind::PERMUTATION) {
          throw std::invalid_argument("VariableImportanceKind::PERMUTATION not supported for a single Tree");
        }

        Tree<T, R, D> std_tree = retrain(center(descale(*training_data)));

        double factor = 1.0;

        if (importance_kind == VariableImportanceKind::PROJECTOR_ADJUSTED) {
          factor = 1.0 / (double)(root->partition_count());
        }

        return std_tree.root->variable_importance(importance_kind) * factor;
      }
  };

  template<typename T, typename R, typename D>
  Tree<T, R, D> train(
    const TrainingSpec<T, R> &training_spec,
    const D &                 training_data,
    std::mt19937&             rng);

  template<typename T, typename R, typename D >
  Tree<T, R, D> train(
    const TrainingSpec<T, R> &training_spec,
    const D &                 training_data);

  template<typename T, typename R, typename D>
  std::ostream& operator<<(std::ostream & ostream, const Tree<T, R, D>& tree) {
    return ostream << tree.to_json().dump(2, ' ', false);
  }

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Node<T, R> &node) {
    return ostream << node.to_json().dump(2, ' ', false);
  }

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Condition<T, R>& condition) {
    return ostream << condition.to_json().dump(2, ' ', false);
  }

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Response<T, R>& response) {
    return ostream << response.to_json().dump(2, ' ', false);
  }
}
