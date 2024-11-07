#include <Rcpp.h>
#include <RcppEigen.h>
#include "pptree.hpp"
#include <random>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace RcppEigen;
using namespace pptree;

// [[Rcpp::export]]
Tree<double, int> pptree_train_glda(
  const Data<double> &    data,
  const DataColumn<int> & groups,
  const double            lambda) {
  return Tree<double, int>::train(
    *TrainingSpec<double, int>::glda(lambda),
    DataSpec<double, int>(data, groups));
}

// [[Rcpp::export]]
Forest<double, int> pptree_train_forest_glda(
  const Data<double> &    data,
  const DataColumn<int> & groups,
  const int               size,
  const int               n_vars,
  const double            lambda,
  SEXP                    n_threads) {
  if (n_threads == R_NilValue) {
    return Forest<double, int>::train(
      *TrainingSpec<double, int>::uniform_glda(n_vars, lambda),
      DataSpec<double, int>(data, groups),
      size,
      R::runif(0, INT_MAX));
  }

  return Forest<double, int>::train(
    *TrainingSpec<double, int>::uniform_glda(n_vars, lambda),
    DataSpec<double, int>(data, groups),
    size,
    R::runif(0, INT_MAX),
    as<const int>(n_threads));
}

// [[Rcpp::export]]
DataColumn<int> pptree_predict(
  const Tree<double, int> &tree,
  const Data<double> &     data) {
  return tree.predict(data);
}

// [[Rcpp::export]]
DataColumn<int> pptree_predict_forest(
  const Forest<double, int> &forest,
  const Data<double> &       data) {
  return forest.predict(data);
}

// [[Rcpp::export]]
Data<double> pptree_variable_importance(
  const Tree<double, int> &tree) {
  return tree.variable_importance(VIProjectorStrategy<double, int>());
}

// [[Rcpp::export]]
Data<double> pptree_forest_variable_importance(
  const Forest<double, int> &forest) {
  DataColumn<double> projector = forest.variable_importance(VIProjectorStrategy<double, int>());
  DataColumn<double> projector_adjusted = forest.variable_importance(VIProjectorAdjustedStrategy<double, int>());
  DataColumn<double> permutation = forest.variable_importance(VIPermutationStrategy<double, int>());

  Data<double> result = Data<double>(projector.size(), 3);

  result.col(0) = projector;
  result.col(1) = projector_adjusted;
  result.col(2) = permutation;

  return result;
}

Data<double> parse_confusion_matrix(const ConfusionMatrix &confusion_matrix) {
  Data<int> values = confusion_matrix.values;
  Data<double> result(values.rows() + 1, values.cols() + 1);

  DataColumn<double> class_errors = confusion_matrix.class_errors();

  for (int i = 0; i < values.rows(); i++) {
    for (int j = 0; j < values.cols(); j++) {
      result(i, j) = values(i, j);
    }
  }

  for (int i = 0; i < values.rows(); i++) {
    result(i, values.cols()) = class_errors(i);
  }

  result(values.rows(), values.cols()) = confusion_matrix.error();

  return result;
}

// [[Rcpp::export]]
Data<double> pptree_confusion_matrix(
  const Tree<double, int> &tree) {
  return parse_confusion_matrix(tree.confusion_matrix(*tree.training_data));
}

// [[Rcpp::export]]
Data<double> pptree_forest_confusion_matrix(
  const Forest<double, int> &forest) {
  return parse_confusion_matrix(forest.confusion_matrix());
}
