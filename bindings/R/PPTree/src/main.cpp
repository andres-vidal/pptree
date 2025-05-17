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
  Data<float>     x,
  DataColumn<int> y,
  const float     lambda) {
  const int seed = R::runif(0, INT_MAX);

  RNG rng(seed);

  return Tree<float, int>::train(
    TrainingSpecGLDA<float, int>(lambda),
    x,
    y,
    rng);
}

// [[Rcpp::export]]
Forest<float, int> pptree_train_forest_glda(
  Data<float>     x,
  DataColumn<int> y,
  const int       size,
  const int       n_vars,
  const float     lambda,
  SEXP            n_threads) {
  const int seed = R::runif(0, INT_MAX);

  if (n_threads == R_NilValue) {
    return Forest<float, int>::train(
      TrainingSpecUGLDA<float, int>(n_vars, lambda),
      x,
      y,
      size,
      seed);
  }

  return Forest<float, int>::train(
    TrainingSpecUGLDA<float, int>(n_vars, lambda),
    x,
    y,
    size,
    seed,
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
