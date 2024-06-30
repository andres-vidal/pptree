#include <Rcpp.h>
#include <RcppEigen.h>
#include "pptree.hpp"
#include <random>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace RcppEigen;
using namespace pptree;

// [[Rcpp::export]]
Tree<long double, int> pptree_train_glda(
  const Data<long double> &data,
  const DataColumn<int> &  groups,
  const double             lambda,
  const int                max_retries) {
  return Tree<long double, int>::train(
    *TrainingSpec<long double, int>::glda(lambda, max_retries),
    DataSpec<long double, int>(data, groups));
}

// [[Rcpp::export]]
Forest<long double, int> pptree_train_forest_glda(
  const Data<long double> & data,
  const DataColumn<int> &   groups,
  const int                 size,
  const int                 n_vars,
  const double              lambda,
  const int                 max_retries) {
  return Forest<long double, int>::train(
    *TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, max_retries),
    DataSpec<long double, int>(data, groups),
    size,
    R::runif(0, INT_MAX));
}

// [[Rcpp::export]]
DataColumn<int> pptree_predict(
  const Tree<long double, int> &tree,
  const Data<long double> &     data) {
  return tree.predict(data);
}

// [[Rcpp::export]]
DataColumn<int> pptree_predict_forest(
  const Forest<long double, int> &forest,
  const Data<long double> &       data) {
  return forest.predict(data);
}

// [[Rcpp::export]]
Data<long double> pptree_variable_importance(
  const Tree<long double, int> &tree) {
  return tree.variable_importance(VIProjectorStrategy<long double, int>());
}

// [[Rcpp::export]]
Data<long double> pptree_forest_variable_importance(
  const Forest<long double, int> &forest) {
  DataColumn<long double> projector = forest.variable_importance(VIProjectorStrategy<long double, int>());
  DataColumn<long double> projector_adjusted = forest.variable_importance(VIProjectorAdjustedStrategy<long double, int>());
  DataColumn<long double> permutation = forest.variable_importance(VIPermutationStrategy<long double, int>());

  Data<long double> result = Data<long double>(projector.size(), 3);

  result.col(0) = projector;
  result.col(1) = projector_adjusted;
  result.col(2) = permutation;

  return result;
}
