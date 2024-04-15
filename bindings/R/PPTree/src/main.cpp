#include <Rcpp.h>
#include <RcppEigen.h>
#include "pptree.hpp"
#include <random>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace RcppEigen;

// [[Rcpp::export]]
pptree::Tree<long double, int> pptree_train_glda(
  pptree::Data<long double> &data,
  pptree::DataColumn<int> &  groups,
  double                     lambda) {
  return pptree::train(
    pptree::TrainingSpec<long double, int>::lda(),
    pptree::DataSpec<long double, int>(data, groups));
}

// [[Rcpp::export]]
pptree::Forest<long double, int> pptree_train_forest_glda(
  const pptree::Data<long double> & data,
  const pptree::DataColumn<int> &   groups,
  const int                         size,
  const int                         n_vars,
  const double                      lambda) {
  return pptree::train(
    pptree::TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
    pptree::DataSpec<long double, int>(data, groups),
    size,
    R::rnorm(0, 1));
}

// [[Rcpp::export]]
pptree::DataColumn<int> pptree_predict(
  pptree::Tree<long double, int> &tree,
  pptree::Data<long double> &     data) {
  return tree.predict(data);
}

// [[Rcpp::export]]
pptree::DataColumn<int> pptree_predict_forest(
  pptree::Forest<long double, int> &forest,
  pptree::Data<long double> &       data) {
  return forest.predict(data);
}

// [[Rcpp::export]]
pptree::Projector<long double> pptree_variable_importance(
  const pptree::Tree<long double, int> &tree) {
  return tree.variable_importance();
}

// [[Rcpp::export]]
pptree::Projector<long double> pptree_forest_variable_importance(
  const pptree::Forest<long double, int> &forest) {
  return forest.variable_importance();
}
