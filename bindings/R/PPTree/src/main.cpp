#include <Rcpp.h>
#include <RcppEigen.h>
#include "pptree.hpp"
#include <random>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace RcppEigen;
using namespace pptree;

// [[Rcpp::export]]
Tree<float, int> pptree_train_glda(
  const Data<float> &     data,
  const DataColumn<int> & groups,
  const float             lambda) {
  return Tree<float, int>::train(
    *TrainingSpec<float, int>::glda(lambda),
    SortedDataSpec<float, int>(data, groups));
}

// [[Rcpp::export]]
Forest<float, int> pptree_train_forest_glda(
  const Data<float> &     data,
  const DataColumn<int> & groups,
  const int               size,
  const int               n_vars,
  const float             lambda,
  SEXP                    n_threads) {
  if (n_threads == R_NilValue) {
    return Forest<float, int>::train(
      *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
      SortedDataSpec<float, int>(data, groups),
      size,
      R::runif(0, INT_MAX));
  }

  return Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    size,
    R::runif(0, INT_MAX),
    as<const int>(n_threads));
}

// [[Rcpp::export]]
DataColumn<int> pptree_predict(
  const Tree<float, int> &tree,
  const Data<float> &     data) {
  return tree.predict(data);
}

// [[Rcpp::export]]
DataColumn<int> pptree_predict_forest(
  const Forest<float, int> &forest,
  const Data<float> &       data) {
  return forest.predict(data);
}

// [[Rcpp::export]]
Data<float> pptree_variable_importance(
  const Tree<float, int> &tree) {
  return tree.variable_importance(VIProjectorStrategy<float, int>());
}

// [[Rcpp::export]]
Data<float> pptree_forest_variable_importance(
  const Forest<float, int> &forest) {
  DataColumn<float> projector          = forest.variable_importance(VIProjectorStrategy<float, int>());
  DataColumn<float> projector_adjusted = forest.variable_importance(VIProjectorAdjustedStrategy<float, int>());
  DataColumn<float> permutation        = forest.variable_importance(VIPermutationStrategy<float, int>());

  Data<float> result = Data<float>(projector.size(), 3);

  result.col(0) = projector;
  result.col(1) = projector_adjusted;
  result.col(2) = permutation;

  return result;
}

Data<float> parse_confusion_matrix(const ConfusionMatrix &confusion_matrix) {
  Data<int> values = confusion_matrix.values;
  Data<float> result(values.rows() + 1, values.cols() + 1);

  DataColumn<float> class_errors = confusion_matrix.class_errors();

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
Data<float> pptree_confusion_matrix(
  const Tree<float, int> &tree) {
  return parse_confusion_matrix(tree.confusion_matrix(tree.training_data));
}

// [[Rcpp::export]]
Data<float> pptree_forest_confusion_matrix(
  const Forest<float, int> &forest) {
  return parse_confusion_matrix(forest.confusion_matrix());
}
