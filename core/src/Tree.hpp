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

  template<typename T, typename R>
  struct Node {
    virtual ~Node() = default;
    virtual R predict(const stats::DataColumn<T> &data) const = 0;
    virtual bool is_response() const = 0;
    virtual bool is_condition() const = 0;
    virtual const Condition<T, R>& as_condition() const = 0;
    virtual const Response<T, R>& as_response() const = 0;

    bool operator==(const Node<T, R> &other) const {
      if (this->is_condition() && other.is_condition()) {
        return this->as_condition() == other.as_condition();
      } else if (this->is_response() && other.is_response()) {
        return this->as_response() == other.as_response();
      } else {
        return false;
      }
    }

    bool operator!=(const Node<T, R> &other) const {
      return !(*this == other);
    }

    virtual std::tuple<pp::Projector<T>, std::set<R> > _variable_importance(const int nvars) const = 0;
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

    R predict(const stats::DataColumn<T> &data) const override {
      T projected_data = pp::project(data, projector);

      if (projected_data < threshold) {
        return lower->predict(data);
      } else {
        return upper->predict(data);
      }
    }

    bool is_response() const override {
      return false;
    }

    bool is_condition() const override {
      return true;
    }

    const Condition<T, R>& as_condition() const override {
      return *this;
    }

    const Response<T, R>& as_response() const override {
      throw std::runtime_error("Cannot cast condition to response");
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

    std::tuple<pp::Projector<T>, std::set<R> > _variable_importance(const int nvars) const override {
      auto [lower_importance, lower_classes] = lower->_variable_importance(nvars);
      auto [upper_importance, upper_classes] = upper->_variable_importance(nvars);

      std::set<R> classes;
      classes.insert(lower_classes.begin(), lower_classes.end());
      classes.insert(upper_classes.begin(), upper_classes.end());

      pp::Projector<T> importance = math::abs(projector) / classes.size();

      return { importance + lower_importance + upper_importance, classes };
    }
  };

  template<typename T, typename R>
  struct Response : public Node<T, R> {
    R value;

    explicit Response(R value) : value(value) {
    }

    R predict(const stats::DataColumn<T> &data) const override {
      return value;
    }

    bool is_response() const override {
      return true;
    }

    bool is_condition() const override {
      return false;
    }

    const Condition<T, R>& as_condition() const override {
      throw std::runtime_error("Cannot cast response to condition");
    }

    const Response<T, R>& as_response() const override {
      return *this;
    }

    bool operator==(const Response<T, R> &other) const {
      return value == other.value;
    }

    bool operator!=(const Response<T, R> &other) const {
      return !(*this == other);
    }

    std::tuple<pp::Projector<T>, std::set<R> > _variable_importance(const int nvars) const override {
      pp::Projector<T> importance = pp::Projector<T>::Zero(nvars);
      return { importance, { value } };
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

    pp::Projector<T> variable_importance() const {
      Tree<T, R, D> std_tree = retrain(center(descale(*training_data)));

      auto [importance, _] = std_tree.root->_variable_importance(training_data->x.cols());

      return importance;
    }

    virtual double error_rate(const stats::DataSpec<T, R> &data) const {
      auto [x, y, _classes] = data.unwrap();
      return stats::error_rate(predict(x), y);
    }

    virtual stats::ConfusionMatrix confusion_matrix(const stats::DataSpec<T, R> &data) const {
      auto [x, y, _classes] = data.unwrap();
      return stats::ConfusionMatrix(predict(x), y);
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


  template<typename T, typename R >
  void to_json(json& j, const Condition<T, R> &condition);
  template<typename T, typename R >
  void to_json(json& j, const Response<T, R> &response);
  template<typename T, typename R >
  void to_json(json& j, const Node<T, R> &node);

  template<typename T, typename R >
  void to_json(json& j, const Condition<T, R>& condition) {
    j = json{
      { "projector", condition.projector },
      { "threshold", condition.threshold },
      { "lower", *condition.lower },
      { "upper", *condition.upper }
    };
  }

  template<typename T, typename R >
  void to_json(json& j, const Response<T, R>& response) {
    j = json{
      { "value", response.value }
    };
  }

  template<typename T, typename R >
  void to_json(json& j, const Node<T, R>& node) {
    if (node.is_response()) {
      to_json(j, node.as_response());
    } else {
      to_json(j, node.as_condition());
    }
  }

  template<typename T, typename R, typename D>
  void to_json(json& j, const Tree<T, R, D>& tree) {
    j = json{
      { "root", *tree.root }
    };
  }

  template<typename T, typename R, typename D>
  std::ostream& operator<<(std::ostream & ostream, const Tree<T, R, D>& tree) {
    json json_tree(tree);
    return ostream << json_tree.dump(2, ' ', false);
  }

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Node<T, R> &node) {
    json json_node(node);
    return ostream << json_node.dump(2, ' ', false);
  }

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Condition<T, R>& condition) {
    json json_condition(condition);
    return ostream << json_condition.dump(2, ' ', false);
  }

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Response<T, R>& response) {
    json json_response(response);
    return ostream << json_response.dump(2, ' ', false);
  }
}
