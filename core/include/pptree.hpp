#include "BootstrapDataSpec.hpp"
#include "DataSpec.hpp"
#include "Group.hpp"
#include "Math.hpp"
#include "Projector.hpp"
#include "PPStrategy.hpp"
#include "DRStrategy.hpp"

#include <memory>
#include <stdexcept>
#include <algorithm>

namespace pptree {
  struct ITrainingParam {
    virtual ~ITrainingParam() = default;
    virtual std::unique_ptr<ITrainingParam> clone() const = 0;
  };

  template<typename T>
  struct TrainingParam : public ITrainingParam {
    T value;

    explicit TrainingParam(const T value) : value(value) {
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
      return ptr;
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
    void set(const std::string &name, T param) {
      map[name] = std::make_unique<TrainingParam<T> >(param);
    }

    template<typename T>
    void set_ptr(const std::string &name, std::shared_ptr<T> param_ptr) {
      map[name] = std::make_unique<TrainingParamPointer<T> >(param_ptr);
    }

    template<typename T>
    T at(const std::string &name) const {
      if (map.find(name) == map.end()) {
        throw std::out_of_range("Parameter " + name + " not found");
      }

      auto ptr = dynamic_cast<const TrainingParam<T> *>(map.at(name).get());

      if (ptr == nullptr) {
        throw std::out_of_range("Parameter '" + name + "' is not of expected type");
      }

      return ptr->value;
    }

    template<typename T>
    T & from_ptr_at(const std::string& name) const {
      if (map.find(name) == map.end()) {
        throw std::out_of_range("Parameter " + name + " not found");
      }

      auto ptr = dynamic_cast<TrainingParamPointer<T> *>(map.at(name).get());

      if (ptr == nullptr) {
        throw std::out_of_range("Parameter '" + name + "' is not of expected type");
      }

      return *ptr->ptr;
    }
  };

  template<typename T, typename R>
  struct TrainingSpec {
    PPStrategy<T, R> pp_strategy;
    DRStrategy<T> dr_strategy;
    std::unique_ptr<TrainingParams> params;

    TrainingSpec(
      const PPStrategy<T, R> pp_strategy,
      const DRStrategy<T>    dr_strategy)
      : pp_strategy(pp_strategy),
      dr_strategy(dr_strategy),
      params(std::make_unique<TrainingParams>()) {
    }

    TrainingSpec(const TrainingSpec& other)
      : pp_strategy(other.pp_strategy),
      dr_strategy(other.dr_strategy),
      params(new TrainingParams(*other.params)) {
    }

    static TrainingSpec<T, R> glda(const double lambda) {
      auto training_spec = TrainingSpec<T, R>(::glda<T, R>(lambda), all<T>());
      training_spec.params->set("lambda", lambda);
      return training_spec;
    }

    static TrainingSpec<T, R> lda() {
      return TrainingSpec<T, R>::glda(0.0);
    }

    static TrainingSpec<T, R> uniform_glda(const int n_vars, const double lambda) {
      auto training_spec = TrainingSpec<T, R>(::glda<T, R>(lambda), uniform<T>(n_vars));
      training_spec.params->set("n_vars", n_vars);
      training_spec.params->set("lambda", lambda);
      return training_spec;
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

  template<typename T, typename R>
  using BootstrapTree = Tree<T, R, BootstrapDataSpec<T, R> >;


  template<typename T, typename R>
  struct Forest {
    std::vector<std::unique_ptr<BootstrapTree<T, R> > > trees;
    std::unique_ptr<TrainingSpec<T, R> > training_spec;
    std::shared_ptr<DataSpec<T, R> > training_data;
    const double seed = 0.0;

    Forest() {
    }

    Forest(
      std::unique_ptr<TrainingSpec<T, R> > && training_spec,
      std::shared_ptr<DataSpec<T, R> > &&     training_data,
      const double                            seed)
      : training_spec(std::move(training_spec)),
      training_data(training_data),
      seed(seed) {
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

    void add_tree(std::unique_ptr<BootstrapTree<T, R> > tree) {
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

    Forest<T, R> retrain(const DataSpec<T, R> &data) const {
      return train(
        *training_spec,
        data,
        trees.size(),
        seed);
    }

    Projector<T> variable_importance() const {
      struct TreeImportance {
        Projector<T> operator()(Projector<T> acc, const std::unique_ptr<BootstrapTree<T, R> >& tree) {
          return acc + tree->variable_importance();
        }
      };

      Forest<T, R> std_forest = retrain(center(descale(*training_data)));

      Projector<T> importance = std::accumulate(
        std_forest.trees.begin(),
        std_forest.trees.end(),
        Projector<T>(Projector<T>::Zero(training_data->x.cols())),
        TreeImportance());


      return importance.array() / std_forest.trees.size();
    }
  };

  template<typename T, typename R, typename D >
  Tree<T, R, D> train(
    const TrainingSpec<T, R> &training_spec,
    const D &                 training_data);

  template<typename T, typename R >
  Forest<T, R> train(
    const TrainingSpec<T, R> &training_spec,
    const DataSpec<T, R> &    training_data,
    const int                 size,
    const double              seed);
}
