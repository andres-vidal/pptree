#include "pp.hpp"

namespace pptree {
  inline namespace pp { using namespace ::pp; }
  inline namespace stats { using namespace ::stats; }
  template<typename T>
  using Threshold = T;

  template<typename T, typename R >
  struct Node {
    virtual ~Node() = default;
    virtual R response() const = 0;
    virtual R predict(DataColumn<T> data) const = 0;
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

    R response() const override {
      throw std::runtime_error("Condition response is undefined.");
    }

    R predict(DataColumn<T> data) const override {
      T projected_data = project((Data<T>)data, projector).value();

      if (projected_data < threshold) {
        return lower->predict(data);
      } else {
        return upper->predict(data);
      }
    }
  };

  template<typename T, typename R >
  struct Response : public Node<T, R> {
    R value;

    Response(R value) : value(value) {
    }

    R response() const override {
      return value;
    }

    R predict(DataColumn<T> data) const override {
      return response();
    }
  };

  template<typename T, typename R >
  struct Tree {
    Condition<T, R> root;

    Tree(Condition<T, R> root) : root(root) {
    }

    R predict(DataColumn<T> data) const {
      return root.predict(data);
    }

    DataColumn<R> predict(Data<T> data) const {
      DataColumn<R> predictions(data.rows());

      for (int i = 0; i < data.rows(); i++) {
        predictions(i) = predict((DataColumn<T>)data.row(i));
      }

      return predictions;
    }
  };

  template<typename T, typename R>
  Tree<T, R> train(
    stats::Data<T>       data,
    stats::DataColumn<R> groups,
    pp::PPStrategy<T, R> pp_strategy);
}
