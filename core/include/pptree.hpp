#include "pp.hpp"
#include <memory>

namespace pptree {
  inline namespace pp { using namespace ::pp; }
  inline namespace stats { using namespace ::stats; }
  template<typename T>
  using Threshold = T;

  template<typename T, typename R>
  struct Node;
  template<typename T, typename R>
  struct Condition;
  template<typename T, typename R>
  struct Response;

  template<typename T, typename R >
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
  };

  template<typename T, typename R >
  struct Condition : public Node<T, R> {
    Projector<T> projector;
    Threshold<T> threshold;
    Node<T, R> *lower = nullptr;
    Node<T, R> *upper = nullptr;

    Condition(
      Projector<T> projector,
      Threshold<T> threshold,
      Node<T, R> *lower,
      Node<T, R> *upper)
      : projector(projector), threshold(threshold), lower(lower), upper(upper) {
    }

    ~Condition() {
      delete lower;
      delete upper;
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
      return linalg::collinear(projector, other.projector)
      && linalg::is_approx(threshold, other.threshold)
      && *lower == *other.lower
      && *upper == *other.upper;
    }
  };

  template<typename T, typename R >
  struct Response : public Node<T, R> {
    R value;

    Response(R value) : value(value) {
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
  };

  template<typename T, typename R >
  struct Tree {
    Condition<T, R> *root;

    Tree(Condition<T, R> *root) : root(root) {
    }

    ~Tree() {
      delete root;
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

    bool operator==(const Tree<T, R> &other) const {
      return *root == *other.root;
    }
  };

  template<typename T, typename R>
  Tree<T, R> train(
    const stats::Data<T>       &data,
    const stats::DataColumn<R> &groups,
    const pp::PPStrategy<T, R> &pp_strategy);

  template<typename T, typename R>
  Tree<T, R> train_lda(
    stats::Data<T>       data,
    stats::DataColumn<R> groups);
}
