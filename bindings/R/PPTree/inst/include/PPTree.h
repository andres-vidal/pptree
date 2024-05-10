#include "pptree.hpp"

#include <RcppCommon.h>

namespace Rcpp {
  SEXP wrap(const pptree::Node<long double, int> &node);
  SEXP wrap(const pptree::Tree<long double, int> &tree);
  SEXP wrap(const pptree::Response<long double, int> &node);
  SEXP wrap(const pptree::Condition<long double, int> &node);
  SEXP wrap(const pptree::Forest<long double, int> &forest);

  SEXP wrap(const pptree::TrainingSpec<long double, int> &training_spec);
  SEXP wrap(const pptree::TrainingParams &params);
  SEXP wrap(const pptree::ITrainingParam &param);
  SEXP wrap(const pptree::TrainingParam<int> &param);
  SEXP wrap(const pptree::TrainingParam<double> &param);
  SEXP wrap(const pptree::DataSpec<long double, int> &data);

  template<> std::unique_ptr<pptree::Node<long double, int> > as(SEXP);
  template<> pptree::Tree<long double, int> as(SEXP);
  template<> pptree::Response<long double, int> as(SEXP);
  template<> pptree::Condition<long double, int> as(SEXP);
  template<> pptree::Forest<long double, int> as(SEXP);

  template<> pptree::TrainingSpec<long double, int>  as(SEXP);
  template<> pptree::DataSpec<long double, int>  as(SEXP);
}


#include <Rcpp.h>

namespace Rcpp {
  SEXP wrap(const pptree::Node<long double, int>& node) {
    if (node.is_response()) {
      return wrap(node.as_response());
    }

    return wrap(node.as_condition());
  }

  SEXP wrap(const pptree::Response<long double, int> &node) {
    return Rcpp::List::create(
      Rcpp::Named("value") = Rcpp::wrap(node.value));
  }

  SEXP wrap(const pptree::Condition<long double, int> &node) {
    return Rcpp::List::create(
      Rcpp::Named("projector") = Rcpp::wrap(node.projector),
      Rcpp::Named("threshold") = Rcpp::wrap(node.threshold),
      Rcpp::Named("lower") = Rcpp::wrap(*node.lower),
      Rcpp::Named("upper") = Rcpp::wrap(*node.upper));
  }

  SEXP wrap(const pptree::Tree<long double, int> &tree) {
    return Rcpp::List::create(
      Rcpp::Named("trainingSpec") = Rcpp::wrap(*tree.training_spec),
      Rcpp::Named("trainingData") = Rcpp::wrap(*tree.training_data),
      Rcpp::Named("root") = Rcpp::wrap(*tree.root));
  }

  SEXP wrap(const pptree::Forest<long double, int> &forest) {
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

  SEXP wrap(const pptree::TrainingSpec<long double, int> &training_spec) {
    return Rcpp::wrap(*training_spec.params);
  }

  SEXP wrap(const pptree::TrainingParams &params) {
    Rcpp::List rparams;

    for (const auto &[key, value_ptr] : params.map) {
      SEXP rvalue = wrap(*value_ptr);

      if (rvalue != R_NilValue) {
        rparams.push_back(wrap(*value_ptr), key);
      }
    }

    return rparams;
  }

  SEXP wrap(const pptree::ITrainingParam &param) {
    if (dynamic_cast<const pptree::TrainingParam<int> *>(&param)) {
      return wrap(dynamic_cast<const pptree::TrainingParam<int>&>(param));
    } else if (dynamic_cast<const pptree::TrainingParam<double> *>(&param)) {
      return wrap(dynamic_cast<const pptree::TrainingParam<double>&>(param));
    } else {
      return R_NilValue;
    }
  }

  SEXP wrap(const pptree::TrainingParam<int> &param) {
    return wrap(param.value);
  }

  SEXP wrap(const pptree::TrainingParam<double> &param) {
    return wrap(param.value);
  }

  SEXP wrap(const pptree::DataSpec<long double, int> &data) {
    return Rcpp::List::create(
      Rcpp::Named("x") = Rcpp::wrap(data.x),
      Rcpp::Named("y") = Rcpp::wrap(data.y),
      Rcpp::Named("classes") = Rcpp::wrap(data.classes));
  }

  template<> std::unique_ptr<pptree::Node<long double, int> > as(SEXP x) {
    Rcpp::List rnode(x);

    if (rnode.containsElementNamed("value")) {
      auto resp = as<pptree::Response<long double, int> >(x);

      auto resp_ptr = std::make_unique<pptree::Response<long double, int> >(std::move(resp));
      return std::move(resp_ptr);
    }

    auto cond = as<pptree::Condition<long double, int> >(x);
    auto cond_ptr = std::make_unique<pptree::Condition<long double, int> >(std::move(cond));
    return std::move(cond_ptr);
  }

  template<> pptree::Response<long double, int> as(SEXP x) {
    Rcpp::List rresp(x);
    return pptree::Response<long double, int>(Rcpp::as<long double>(rresp["value"]));
  }

  template<> pptree::Condition<long double, int> as(SEXP x) {
    Rcpp::List rcond(x);

    auto lower = as<std::unique_ptr<pptree::Node<long double, int> > >(rcond["lower"]);
    auto upper = as<std::unique_ptr<pptree::Node<long double, int> > >(rcond["upper"]);

    return pptree::Condition<long double, int>(
      Rcpp::as<pptree::Projector<long double> >(rcond["projector"]),
      Rcpp::as<long double>(rcond["threshold"]),
      std::move(lower),
      std::move(upper));
  }

  template<> pptree::Tree<long double, int> as(SEXP x) {
    Rcpp::List rtree(x);
    Rcpp::List rtraining_spec(rtree["trainingSpec"]);
    Rcpp::List rtraining_data(rtree["trainingData"]);

    auto root = as<pptree::Condition<long double, int> >(rtree["root"]);
    auto root_ptr = std::make_unique<pptree::Condition<long double, int> >(std::move(root));

    return pptree::Tree<long double, int>(
      std::move(root_ptr),
      std::make_unique<pptree::TrainingSpec<long double, int> >(as<pptree::TrainingSpec<long double, int> >(rtraining_spec)),
      std::make_shared<pptree::DataSpec<long double, int> >(as<pptree::DataSpec<long double, int> >(rtraining_data)));
  }

  template<> pptree::Forest<long double, int> as(SEXP x) {
    Rcpp::List rforest(x);
    Rcpp::List rtrees(rforest["trees"]);
    Rcpp::List rtraining_spec(rforest["trainingSpec"]);
    Rcpp::List rtraining_data(rforest["trainingData"]);

    pptree::Forest<long double, int> forest(
      std::make_unique<pptree::TrainingSpec<long double, int> >(as<pptree::TrainingSpec<long double, int> >(rtraining_spec)),
      std::make_shared<pptree::DataSpec<long double, int> >(as<pptree::DataSpec<long double, int> >(rtraining_data)),
      Rcpp::as<double>(rforest["seed"]));

    for (size_t i = 0; i < rtrees.size(); i++) {
      auto tree = as<pptree::Tree<long double, int> >(rtrees[i]);
      auto tree_ptr = std::make_unique<pptree::Tree<long double, int> >(std::move(tree));
      forest.add_tree(std::move(tree_ptr));
    }

    return forest;
  }

  template<> pptree::TrainingSpec<long double, int> as(SEXP x) {
    Rcpp::List rspec(x);

    bool has_n_vars = rspec.containsElementNamed("n_vars");
    bool has_lambda = rspec.containsElementNamed("lambda");

    if (has_n_vars && has_lambda) {
      return pptree::TrainingSpec<long double, int>::uniform_glda(
        Rcpp::as<int>(rspec["n_vars"]),
        Rcpp::as<double>(rspec["lambda"]));
    }

    if (has_lambda) {
      return pptree::TrainingSpec<long double, int>::glda(
        Rcpp::as<double>(rspec["lambda"]));
    }

    Rcpp::stop("Invalid training spec");
  }

  template<> pptree::DataSpec<long double, int> as(SEXP x) {
    Rcpp::List rdata(x);

    std::vector<int> classes = Rcpp::as<std::vector<int> >(rdata["classes"]);

    return pptree::DataSpec<long double, int>(
      Rcpp::as<pptree::Data<long double> >(rdata["x"]),
      Rcpp::as<pptree::DataColumn<int> >(rdata["y"]),
      std::set<int>(classes.begin(), classes.end()));
  }
}
