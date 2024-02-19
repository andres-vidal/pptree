#include "pp.hpp"

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

  template<typename T, typename R>
  Response<T, R> * as_response(Node<T, R> *node);
  template<typename T, typename R>
  const Response<T, R>& as_response(const Node<T, R> &node);
  template<typename T, typename R>
  Condition<T, R> * as_condition(Node<T, R> *node);
  template<typename T, typename R>
  const Condition<T, R> & as_condition(const Node<T, R> &node);


  template<typename T, typename R >
  struct Node {
    virtual ~Node() = default;
    virtual R predict(const DataColumn<T> &data) const = 0;
    virtual bool is_response() const = 0;
    virtual bool is_condition() const = 0;

    bool operator==(const Node<T, R> &other) const {
      if (this->is_condition() && other.is_condition()) {
        return as_condition(*this) == as_condition(other);
      } else if (this->is_response() && other.is_response()) {
        return as_response(*this) == as_response(other);
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
      T projected_data = project((Data<T>)data, projector).value();

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

    bool operator==(const Condition<T, R> &other) const {
      T tolerance = 0.00001;

      return linalg::collinear(projector, other.projector)
      && abs(threshold - other.threshold) < tolerance
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
  Response<T, R> * as_response(Node<T, R> *node) {
    if (Response<T, R> *response = dynamic_cast<Response<T, R> *>(node)) {
      return response;
    }

    throw std::runtime_error("Node is not a response.");
  }

  template<typename T, typename R>
  const Response<T, R>& as_response(const Node<T, R> &node) {
    return dynamic_cast<const Response<T, R> &>(node);
  }

  template<typename T, typename R>
  Condition<T, R> * as_condition(Node<T, R> *node) {
    if (Condition<T, R> *condition = dynamic_cast<Condition<T, R> *>(node)) {
      return condition;
    }

    throw std::runtime_error("Node is not a condition.");
  }

  template<typename T, typename R>
  const Condition<T, R>& as_condition(const Node<T, R> &node) {
    return dynamic_cast<const Condition<T, R> &>(node);
  }
}
