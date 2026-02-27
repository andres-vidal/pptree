#include <Rcpp.h>
#include <RcppEigen.h>
#include "pptree.hpp"
#include <random>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace RcppEigen;
using namespace pptree;

// [[Rcpp::export]]
Tree pptree_train_glda(
  FeatureMatrix  x,
  ResponseVector y,
  const float    lambda) {
  const int seed = R::runif(0, INT_MAX);

  RNG rng(seed);

  return Tree::train(
    TrainingSpecGLDA(lambda),
    x,
    y,
    rng);
}

// [[Rcpp::export]]
Forest pptree_train_forest_glda(
  FeatureMatrix  x,
  ResponseVector y,
  const int      size,
  const int      n_vars,
  const float    lambda,
  SEXP           n_threads) {
  const int seed = R::runif(0, INT_MAX);

  if (n_threads == R_NilValue) {
    return Forest::train(
      TrainingSpecUGLDA(n_vars, lambda),
      x,
      y,
      size,
      seed);
  }

  return Forest::train(
    TrainingSpecUGLDA(n_vars, lambda),
    x,
    y,
    size,
    seed,
    as<const int>(n_threads));
}

// [[Rcpp::export]]
ResponseVector pptree_predict(
  const Tree &          tree,
  const FeatureMatrix & data) {
  return tree.predict(data);
}

// [[Rcpp::export]]
ResponseVector pptree_predict_forest(
  const Forest &        forest,
  const FeatureMatrix & data) {
  return forest.predict(data);
}
