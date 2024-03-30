#include "pp.hpp"
#include "dr.hpp"
#include <memory>
#include <stdexcept>
#include <any>
namespace pptree {
  inline namespace pp { using namespace ::pp; }
  inline namespace dr { using namespace ::dr; }
  inline namespace ppstrategy { using namespace ::pp::strategy; }
  inline namespace drstrategy { using namespace ::dr::strategy; }
  inline namespace stats { using namespace ::stats; }

  template<typename T, typename R>
  struct TrainingSpec {
    const PPStrategy<T, R> pp_strategy;
    const DRStrategy<T> dr_strategy;
    const std::map<std::string, std::any> params;

    TrainingSpec(
      const PPStrategy<T, R> pp_strategy,
      const DRStrategy<T> dr_strategy,
      const std::map<std::string, std::any> params)
      : pp_strategy(pp_strategy), dr_strategy(dr_strategy), params(params) {
    }
  };

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
  };

  template<typename T, typename R>
  struct Condition : public Node<T, R> {
    Projector<T> projector;
    Threshold<T> threshold;
    std::unique_ptr<Node<T, R> > lower;
    std::unique_ptr<Node<T, R> > upper;

    Condition(
      Projector<T> projector,
      Threshold<T> threshold,
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
      return linalg::collinear(projector, other.projector)
      && linalg::is_approx(threshold, other.threshold)
      && *lower == *other.lower
      && *upper == *other.upper;
    }

    bool operator!=(const Condition<T, R> &other) const {
      return !(*this == other);
    }
  };

  template<typename T, typename R>
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

    bool operator!=(const Response<T, R> &other) const {
      return !(*this == other);
    }
  };

  template<typename T, typename R>
  struct Tree {
    std::unique_ptr<Condition<T, R> > root;

    Tree(std::unique_ptr<Condition<T, R> > root) : root(std::move(root)) {
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

    bool operator!=(const Tree<T, R> &other) const {
      return !(*this == other);
    }
  };

  template<typename T, typename R>
  struct Forest {
    std::vector<std::unique_ptr<Tree<T, R> > > trees;
    const int n_vars;
    const double seed;
    const double lambda;


    Forest(const int n_vars, const double lambda, const double seed)
      : n_vars(n_vars),  lambda(lambda), seed(seed) {
    }

    R predict(const DataColumn<T> &data) const {
      std::map<R, int> votes_per_group;

      for (const auto &tree : trees) {
        R prediction = tree->predict(data);

        if (votes_per_group.find(prediction) == votes_per_group.end()) {
          votes_per_group[prediction] = 1;
        } else {
          votes_per_group[prediction] += 1;
        }
      }

      int most_voted_group_votes = 0;
      R most_voted_group;

      for (const auto &[key, votes] : votes_per_group) {
        if (votes > most_voted_group_votes) {
          most_voted_group = key;
          most_voted_group_votes = votes;
        }
      }

      return most_voted_group;
    }

    DataColumn<R> predict(const Data<T> &data) const {
      DataColumn<R> predictions(data.rows());

      for (int i = 0; i < data.rows(); i++) {
        predictions(i) = predict((DataColumn<T>)data.row(i));
      }

      return predictions;
    }

    void add_tree(std::unique_ptr<Tree<T, R> > tree) {
      trees.push_back(std::move(tree));
    }

    bool operator==(const Forest<T, R> &other) const {
      if (trees.size() != other.trees.size()) {
        return false;
      }

      for (int i = 0; i < trees.size(); i++) {
        if (*trees[i] != *other.trees[i]) {
          return false;
        }
      }

      return true;
    }

    bool operator!=(const Forest<T, R> &other) const {
      return !(*this == other);
    }
  };

  template<typename T, typename R>
  Tree<T, R> train_glda(
    const stats::Data<T> &      data,
    const stats::DataColumn<R> &groups,
    const double lambda);

  template<typename T, typename R>
  Forest<T, R> train_forest_glda(
    const Data<T> &         data,
    const DataColumn<R> &   groups,
    const int size,
    const int n_vars,
    const double lambda,
    const double seed);
}
