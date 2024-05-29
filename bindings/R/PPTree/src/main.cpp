#include <Rcpp.h>
#include <RcppEigen.h>
#include "pptree.hpp"
#include <random>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace RcppEigen;

// [[Rcpp::export]]
Tree<long double, int> pptree_train_glda(
  Data<long double> &data,
  DataColumn<int> &  groups,
  double             lambda) {
  return train(
    TrainingSpec<long double, int>::lda(),
    DataSpec<long double, int>(data, groups));
}

// [[Rcpp::export]]
Forest<long double, int> pptree_train_forest_glda(
  const Data<long double> & data,
  const DataColumn<int> &   groups,
  const int                 size,
  const int                 n_vars,
  const double              lambda) {
  return train(
    TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
    DataSpec<long double, int>(data, groups),
    size,
    R::rnorm(0, 1));
}

// [[Rcpp::export]]
DataColumn<int> pptree_predict(
  Tree<long double, int> &tree,
  Data<long double> &     data) {
  return tree.predict(data);
}

// [[Rcpp::export]]
DataColumn<int> pptree_predict_forest(
  Forest<long double, int> &forest,
  Data<long double> &       data) {
  return forest.predict(data);
}

// [[Rcpp::export]]
Projector<long double> pptree_variable_importance(
  const Tree<long double, int> &tree) {
  return tree.variable_importance();
}

// [[Rcpp::export]]
Projector<long double> pptree_forest_variable_importance(
  const Forest<long double, int> &forest) {
  return forest.variable_importance();
}
