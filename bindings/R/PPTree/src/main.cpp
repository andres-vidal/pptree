#include <Rcpp.h>
#include <RcppEigen.h>
#include "pptree.hpp"
#include <random>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace RcppEigen;
using namespace pptree;
using namespace pptree::types;
using namespace pptree::stats;

// [[Rcpp::export]]
Tree pptree_train_glda(
  FeatureMatrix  x,
  ResponseVector y,
  const float    lambda) {
  const int seed = R::runif(0, INT_MAX);

  if (!GroupPartition::is_contiguous(y)) {
    sort(x, y);
  }

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

  if (!GroupPartition::is_contiguous(y)) {
    sort(x, y);
  }

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

// [[Rcpp::export]]
FeatureVector pptree_vi_projections_tree(
  const Tree&          tree,
  int                  n_vars,
  const FeatureVector& scale) {
  return variable_importance_projections(tree, n_vars, &scale);
}

// [[Rcpp::export]]
FeatureVector pptree_vi_projections_forest(
  const Forest&        forest,
  int                 n_vars,
  const FeatureVector& scale) {
  return variable_importance_projections(forest, n_vars, &scale);
}

// [[Rcpp::export]]
FeatureVector pptree_vi_weighted(
  const Forest&        forest,
  const FeatureMatrix& x,
  const ResponseVector& y,
  const FeatureVector& scale) {
  return variable_importance_weighted_projections(forest, x, y, &scale);
}

// [[Rcpp::export]]
FeatureVector pptree_vi_permuted(
  const Forest&        forest,
  const FeatureMatrix& x,
  const ResponseVector& y,
  int                  seed) {
  return variable_importance_permuted(forest, x, y, seed);
}

// [[Rcpp::export]]
double pptree_oob_error(
  const Forest&        forest,
  const FeatureMatrix& x,
  const ResponseVector& y) {
  return forest.oob_error(x, y);
}
