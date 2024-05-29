#pragma once

#include "DataSpec.hpp"
#include "TrainingSpec.hpp"
#include "Projector.hpp"

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
  virtual R predict(const DataColumn<T> &data) const = 0;
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

  virtual std::tuple<Projector<T>, std::set<R> > _variable_importance(const int nvars) const = 0;
};

template<typename T, typename R>
struct Condition : public Node<T, R> {
  Projector<T> projector;
  Threshold<T> threshold;
  std::unique_ptr<Node<T, R> > lower;
  std::unique_ptr<Node<T, R> > upper;

  Condition(
    Projector<T>                 projector,
    Threshold<T>                 threshold,
    std::unique_ptr<Node<T, R> > lower,
    std::unique_ptr<Node<T, R> > upper)
    : projector(projector), threshold(threshold), lower(std::move(lower)), upper(std::move(upper)) {
  }

  R predict(const DataColumn<T> &data) const override {
    T projected_data = project(data, projector);

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
    return collinear(projector, other.projector)
           && is_approx(threshold, other.threshold)
           && *lower == *other.lower
           && *upper == *other.upper;
  }

  bool operator!=(const Condition<T, R> &other) const {
    return !(*this == other);
  }

  std::tuple<Projector<T>, std::set<R> > _variable_importance(const int nvars) const override {
    auto [lower_importance, lower_classes] = lower->_variable_importance(nvars);
    auto [upper_importance, upper_classes] = upper->_variable_importance(nvars);

    std::set<R> classes;
    classes.insert(lower_classes.begin(), lower_classes.end());
    classes.insert(upper_classes.begin(), upper_classes.end());

    Projector<T> importance = abs(projector) / classes.size();

    return { importance + lower_importance + upper_importance, classes };
  }
};

template<typename T, typename R>
struct Response : public Node<T, R> {
  R value;

  explicit Response(R value) : value(value) {
  }

  R predict(const DataColumn<T> &data) const override {
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

  std::tuple<Projector<T>, std::set<R> > _variable_importance(const int nvars) const override {
    Projector<T> importance = Projector<T>::Zero(nvars);
    return { importance, { value } };
  }
};

template<typename T, typename R, typename D = DataSpec<T, R> >
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

  R predict(const DataColumn<T> &data) const {
    return root->predict(data);
  }

  DataColumn<R> predict(const Data<T> &data) const {
    DataColumn<R> predictions(data.rows());

    for (int i = 0; i < data.rows(); i++) {
      predictions(i) = predict((DataColumn<T>)data.row(i));
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

  Projector<T> variable_importance() const {
    Tree<T, R, D> std_tree = retrain(center(descale(*training_data)));

    auto [importance, _] = std_tree.root->_variable_importance(training_data->x.cols());

    return importance;
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
