#include "pptree.hpp"
#include <iostream>

#include <RcppCommon.h>

namespace Rcpp {
  SEXP wrap(const pptree::Node<long double, int> &node);
  SEXP wrap(const pptree::Tree<long double, int> &tree);
  SEXP wrap(const pptree::Response<long double, int> &node);
  SEXP wrap(const pptree::Condition<long double, int> &node);

  template<> std::unique_ptr<pptree::Node<long double, int> > as(SEXP);
  template<> pptree::Tree<long double, int> as(SEXP);
  template<> pptree::Response<long double, int> as(SEXP);
  template<> pptree::Condition<long double, int> as(SEXP);
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
      Rcpp::Named("root") = Rcpp::wrap(*tree.root));
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

    auto root = as<pptree::Condition<long double, int> >(rtree["root"]);
    auto root_ptr = std::make_unique<pptree::Condition<long double, int> >(std::move(root));

    return pptree::Tree<long double, int>(std::move(root_ptr));
  }
}
