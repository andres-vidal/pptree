#pragma once

#include "Data.hpp"
#include "DataColumn.hpp"
#include "SortedDataSpec.hpp"
#include "Projector.hpp"
#include "TrainingSpec.hpp"

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
    virtual json to_json() const = 0;
    virtual bool equals(const Node<T, R> &other) const = 0;
    virtual bool equals(const Condition<T, R> &other) const = 0;
    virtual bool equals(const Response<T, R> &other) const = 0;


    bool operator==(const Node<T, R> &other) const {
      return this->equals(other);
    }

    bool operator!=(const Node<T, R> &other) const {
      return !this->equals(other);
    }
  };

  template<typename T, typename R>
  struct Condition : public Node<T, R> {
    pp::Projector<T> projector;
    T threshold;
    std::unique_ptr<Node<T, R> > lower;
    std::unique_ptr<Node<T, R> > upper;
    std::unique_ptr<TrainingSpec<T, R> > training_spec;
    std::unique_ptr<stats::SortedDataSpec<T, R> > training_data;

    Condition(
      const pp::Projector<T>&                       projector,
      const Threshold<T>&                           threshold,
      std::unique_ptr<Node<T, R> >                  lower,
      std::unique_ptr<Node<T, R> >                  upper,
      std::unique_ptr<TrainingSpec<T, R> >          training_spec,
      std::unique_ptr<stats::SortedDataSpec<T, R> > training_data)
      : projector(projector),
      threshold(threshold),
      lower(std::move(lower)),
      upper(std::move(upper)),
      training_spec(std::move(training_spec)),
      training_data(std::move(training_data)) {
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
      T projected_data = pp::project(data.transpose(), projector).value();

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
  };

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
