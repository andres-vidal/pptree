#include "pp.hpp"
#include "dr.hpp"
#include <memory>
#include <stdexcept>
#include <algorithm>

namespace pptree {
  inline namespace pp { using namespace ::pp; }
  inline namespace dr { using namespace ::dr; }
  inline namespace ppstrategy { using namespace ::pp::strategy; }
  inline namespace drstrategy { using namespace ::dr::strategy; }
  inline namespace stats { using namespace ::stats; }


  struct ITrainingParam {
    virtual ~ITrainingParam() = default;
    virtual std::unique_ptr<ITrainingParam> clone() const = 0;
  };

  template<typename T>
  struct TrainingParam : public ITrainingParam {
    T value;

    TrainingParam(const T value) : value(value) {
    }

    std::unique_ptr<ITrainingParam> clone() const override {
      return std::make_unique<TrainingParam<T> >(value);
    }
  };

  template<typename T>
  struct TrainingParamPointer : public ITrainingParam {
    std::shared_ptr<T> ptr;

    explicit TrainingParamPointer(std::shared_ptr<T> ptr) : ptr(ptr) {
    }

    std::unique_ptr<ITrainingParam> clone() const override {
      return std::make_unique<TrainingParamPointer<T> >(ptr);
    }
  };


  struct TrainingParams {
    std::map<std::string, std::unique_ptr<ITrainingParam> > map;

    TrainingParams() {
    }

    TrainingParams(const TrainingParams& other) {
      for (const auto& [key, value] : other.map) {
        map[key] = value->clone();
      }
    }

    template<typename T>
    void set(const std::string name, T param) {
      map[name] = std::make_unique<TrainingParam<T> >(param);
    }

    template<typename T>
    void set_ptr(const std::string name, std::shared_ptr<T> param_ptr) {
      map[name] = std::make_unique<TrainingParamPointer<T> >(param_ptr);
    }

    template<typename T>
    T at(const std::string name) const {
      return dynamic_cast<TrainingParam<T> *>(map.at(name).get())->value;
    }

    template<typename T>
    T & from_ptr_at(const std::string name) const {
      return *dynamic_cast<TrainingParamPointer<T> *>(map.at(name).get())->ptr;
    }
  };

  template<typename T, typename R>
  struct TrainingSpec {
    PPStrategy<T, R> pp_strategy;
    DRStrategy<T> dr_strategy;
    std::unique_ptr<TrainingParams> params;

    TrainingSpec(
      const PPStrategy<T, R> pp_strategy,
      const DRStrategy<T> dr_strategy)
      : pp_strategy(pp_strategy), dr_strategy(dr_strategy), params(std::make_unique<TrainingParams>()) {
    }

    TrainingSpec(const TrainingSpec& other)
      : pp_strategy(other.pp_strategy), dr_strategy(other.dr_strategy),
        params(new TrainingParams(*other.params)) {
    }

    std::unique_ptr<TrainingSpec<T, R> > clone() const {
      return std::make_unique<TrainingSpec<T, R> >(*this);
    }

    static std::unique_ptr<TrainingSpec<T, R> > glda(const double lambda) {
      auto spec = std::make_unique<TrainingSpec<T, R> >(pp::strategy::glda<T, R>(lambda), dr::strategy::all<T>());

      spec->params->set("lambda", lambda);
      return spec;
    }

    static std::unique_ptr<TrainingSpec<T, R> > lda() {
      return TrainingSpec<T, R>::glda(0.0);
    }


    static std::unique_ptr<TrainingSpec<T, R> > uniform_glda(const int n_vars, const double lambda, const double seed) {
      auto rng =  std::make_shared<std::mt19937>(seed);
      auto spec = std::make_unique<TrainingSpec<T, R> >(pp::strategy::glda<T, R>(lambda), dr::strategy::uniform<T>(n_vars, *rng));
      spec->params->set("n_vars", n_vars);
      spec->params->set("lambda", lambda);
      spec->params->set("seed", seed);
      spec->params->set_ptr("rng", rng);
      return spec;
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
    std::unique_ptr<TrainingSpec<T, R> > spec;

    Tree(std::unique_ptr<Condition<T, R> > root,  std::unique_ptr<TrainingSpec<T, R> > spec) : root(std::move(root)), spec(std::move(spec)) {
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
    std::unique_ptr<TrainingSpec<T, R> > spec;

    Forest(std::unique_ptr<TrainingSpec<T, R> > && spec) : spec(std::move(spec)) {
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
