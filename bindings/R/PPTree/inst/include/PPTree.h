#include "pptree.hpp"

#include <RcppCommon.h>

using namespace pptree;
using namespace pptree::stats;
using namespace pptree::pp;

namespace Rcpp {
  SEXP wrap(const Node<long double, int> &node);
  SEXP wrap(const Tree<long double, int, DataSpec<long double, int> > &tree);
  SEXP wrap(const Tree<long double, int, BootstrapDataSpec<long double, int> > &tree);
  SEXP wrap(const Response<long double, int> &node);
  SEXP wrap(const Condition<long double, int> &node);
  SEXP wrap(const Forest<long double, int> &forest);

  SEXP wrap(const TrainingSpec<long double, int> &training_spec);
  SEXP wrap(const TrainingParams &params);
  SEXP wrap(const ITrainingParam &param);
  SEXP wrap(const TrainingParam<int> &param);
  SEXP wrap(const TrainingParam<double> &param);
  SEXP wrap(const DataSpec<long double, int> &data);
  SEXP wrap(const BootstrapDataSpec<long double, int> &data);

  template<> std::unique_ptr<Node<long double, int> > as(SEXP);
  template<> Tree<long double, int, DataSpec<long double, int> > as(SEXP);
  template<> Tree<long double, int, BootstrapDataSpec<long double, int> > as(SEXP);
  template<> Response<long double, int> as(SEXP);
  template<> Condition<long double, int> as(SEXP);
  template<> Forest<long double, int> as(SEXP);

  template<> TrainingSpec<long double, int>  as(SEXP);
  template<> DataSpec<long double, int>  as(SEXP);
  template<> BootstrapDataSpec<long double, int> as(SEXP);
}


#include <Rcpp.h>

namespace Rcpp {
  SEXP wrap(const Node<long double, int>& node) {
    if (node.is_response()) {
      return wrap(node.as_response());
    }

    return wrap(node.as_condition());
  }

  SEXP wrap(const Response<long double, int> &node) {
    return Rcpp::List::create(
      Rcpp::Named("value") = Rcpp::wrap(node.value));
  }

  SEXP wrap(const Condition<long double, int> &node) {
    return Rcpp::List::create(
      Rcpp::Named("projector") = Rcpp::wrap(node.projector),
      Rcpp::Named("threshold") = Rcpp::wrap(node.threshold),
      Rcpp::Named("lower") = Rcpp::wrap(*node.lower),
      Rcpp::Named("upper") = Rcpp::wrap(*node.upper));
  }

  SEXP wrap(const Tree<long double, int, DataSpec<long double, int> > &tree) {
    return Rcpp::List::create(
      Rcpp::Named("trainingSpec") = Rcpp::wrap(*tree.training_spec),
      Rcpp::Named("trainingData") = Rcpp::wrap(*tree.training_data),
      Rcpp::Named("root") = Rcpp::wrap(*tree.root));
  }

  SEXP wrap(const Tree<long double, int, BootstrapDataSpec<long double, int> > &tree) {
    return Rcpp::List::create(
      Rcpp::Named("trainingSpec") = Rcpp::wrap(*tree.training_spec),
      Rcpp::Named("trainingData") = Rcpp::wrap(*tree.training_data),
      Rcpp::Named("root") = Rcpp::wrap(*tree.root));
  }

  SEXP wrap(const Forest<long double, int> &forest) {
    Rcpp::List trees(forest.trees.size());

    for (size_t i = 0; i < forest.trees.size(); i++) {
      trees[i] = wrap(*forest.trees[i]);
    }

    return Rcpp::List::create(
      Rcpp::Named("trainingSpec") = Rcpp::wrap(*forest.training_spec),
      Rcpp::Named("trainingData") = Rcpp::wrap(*forest.training_data),
      Rcpp::Named("seed") = Rcpp::wrap(forest.seed),
      Rcpp::Named("trees") = trees);
  }

  SEXP wrap(const TrainingSpec<long double, int> &training_spec) {
    return Rcpp::wrap(*training_spec.params);
  }

  SEXP wrap(const TrainingParams &params) {
    Rcpp::List rparams;

    for (const auto &[key, value_ptr] : params.map) {
      SEXP rvalue = wrap(*value_ptr);

      if (rvalue != R_NilValue) {
        rparams.push_back(wrap(*value_ptr), key);
      }
    }

    return rparams;
  }

  SEXP wrap(const ITrainingParam &param) {
    if (dynamic_cast<const TrainingParam<int> *>(&param)) {
      return wrap(dynamic_cast<const TrainingParam<int>&>(param));
    } else if (dynamic_cast<const TrainingParam<double> *>(&param)) {
      return wrap(dynamic_cast<const TrainingParam<double>&>(param));
    } else {
      return R_NilValue;
    }
  }

  SEXP wrap(const TrainingParam<int> &param) {
    return wrap(param.value);
  }

  SEXP wrap(const TrainingParam<double> &param) {
    return wrap(param.value);
  }

  SEXP wrap(const DataSpec<long double, int> &data) {
    return Rcpp::List::create(
      Rcpp::Named("x") = Rcpp::wrap(data.x),
      Rcpp::Named("y") = Rcpp::wrap(data.y),
      Rcpp::Named("classes") = Rcpp::wrap(data.classes));
  }

  SEXP wrap(const BootstrapDataSpec<long double, int> &data) {
    return Rcpp::List::create(
      Rcpp::Named("x") = Rcpp::wrap(data.x),
      Rcpp::Named("y") = Rcpp::wrap(data.y),
      Rcpp::Named("classes") = Rcpp::wrap(data.classes),
      Rcpp::Named("sampleIndices") = Rcpp::wrap(data.sample_indices));
  }

  template<> std::unique_ptr<Node<long double, int> > as(SEXP x) {
    Rcpp::List rnode(x);

    if (rnode.containsElementNamed("value")) {
      auto resp = as<Response<long double, int> >(x);

      auto resp_ptr = std::make_unique<Response<long double, int> >(std::move(resp));
      return resp_ptr;
    }

    auto cond = as<Condition<long double, int> >(x);
    auto cond_ptr = std::make_unique<Condition<long double, int> >(std::move(cond));
    return cond_ptr;
  }

  template<> Response<long double, int> as(SEXP x) {
    Rcpp::List rresp(x);
    return Response<long double, int>(Rcpp::as<long double>(rresp["value"]));
  }

  template<> Condition<long double, int> as(SEXP x) {
    Rcpp::List rcond(x);

    auto lower = as<std::unique_ptr<Node<long double, int> > >(rcond["lower"]);
    auto upper = as<std::unique_ptr<Node<long double, int> > >(rcond["upper"]);

    return Condition<long double, int>(
      Rcpp::as<Projector<long double> >(rcond["projector"]),
      Rcpp::as<long double>(rcond["threshold"]),
      std::move(lower),
      std::move(upper));
  }

  template<> Tree<long double, int, DataSpec<long double, int> > as(SEXP x) {
    Rcpp::List rtree(x);
    Rcpp::List rtraining_spec(rtree["trainingSpec"]);
    Rcpp::List rtraining_data(rtree["trainingData"]);

    auto root = as<Condition<long double, int> >(rtree["root"]);
    auto root_ptr = std::make_unique<Condition<long double, int> >(std::move(root));

    return Tree<long double, int, DataSpec<long double, int> >(
      std::move(root_ptr),
      std::make_unique<TrainingSpec<long double, int> >(as<TrainingSpec<long double, int> >(rtraining_spec)),
      std::make_shared<DataSpec<long double, int> >(as<DataSpec<long double, int> >(rtraining_data)));
  }

  template<> Tree<long double, int, BootstrapDataSpec<long double, int> > as(SEXP x) {
    Rcpp::List rtree(x);
    Rcpp::List rtraining_spec(rtree["trainingSpec"]);
    Rcpp::List rtraining_data(rtree["trainingData"]);

    auto root = as<Condition<long double, int> >(rtree["root"]);
    auto root_ptr = std::make_unique<Condition<long double, int> >(std::move(root));

    return Tree<long double, int, BootstrapDataSpec<long double, int> >(
      std::move(root_ptr),
      std::make_unique<TrainingSpec<long double, int> >(as<TrainingSpec<long double, int> >(rtraining_spec)),
      std::make_shared<BootstrapDataSpec<long double, int> >(as<BootstrapDataSpec<long double, int> >(rtraining_data)));
  }

  template<> Forest<long double, int> as(SEXP x) {
    Rcpp::List rforest(x);
    Rcpp::List rtrees(rforest["trees"]);
    Rcpp::List rtraining_spec(rforest["trainingSpec"]);
    Rcpp::List rtraining_data(rforest["trainingData"]);

    Forest<long double, int> forest(
      std::make_unique<TrainingSpec<long double, int> >(as<TrainingSpec<long double, int> >(rtraining_spec)),
      std::make_shared<DataSpec<long double, int> >(as<DataSpec<long double, int> >(rtraining_data)),
      Rcpp::as<double>(rforest["seed"]));

    for (size_t i = 0; i < rtrees.size(); i++) {
      auto tree = as<Tree<long double, int, BootstrapDataSpec<long double, int> > >(rtrees[i]);
      auto tree_ptr = std::make_unique<Tree<long double, int, BootstrapDataSpec<long double, int> > >(std::move(tree));
      forest.add_tree(std::move(tree_ptr));
    }

    return forest;
  }

  template<> TrainingSpec<long double, int> as(SEXP x) {
    Rcpp::List rspec(x);

    bool has_n_vars = rspec.containsElementNamed("n_vars");
    bool has_lambda = rspec.containsElementNamed("lambda");

    if (has_n_vars && has_lambda) {
      return TrainingSpec<long double, int>::uniform_glda(
        Rcpp::as<int>(rspec["n_vars"]),
        Rcpp::as<double>(rspec["lambda"]));
    }

    if (has_lambda) {
      return TrainingSpec<long double, int>::glda(
        Rcpp::as<double>(rspec["lambda"]));
    }

    Rcpp::stop("Invalid training spec");
  }

  template<> DataSpec<long double, int> as(SEXP x) {
    Rcpp::List rdata(x);

    std::vector<int> classes = Rcpp::as<std::vector<int> >(rdata["classes"]);

    return DataSpec<long double, int>(
      Rcpp::as<Data<long double> >(rdata["x"]),
      Rcpp::as<DataColumn<int> >(rdata["y"]),
      std::set<int>(classes.begin(), classes.end()));
  }

  template<> BootstrapDataSpec<long double, int> as(SEXP x) {
    Rcpp::List rdata(x);

    std::vector<int> classes = Rcpp::as<std::vector<int> >(rdata["classes"]);
    std::vector<int> sample_indices = Rcpp::as<std::vector<int> >(rdata["sampleIndices"]);

    return BootstrapDataSpec<long double, int>(
      Rcpp::as<Data<long double> >(rdata["x"]),
      Rcpp::as<DataColumn<int> >(rdata["y"]),
      std::set<int>(classes.begin(), classes.end()),
      sample_indices);
  }
}
