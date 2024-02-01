#include <Rcpp.h>
#include <RcppEigen.h>
#include "pptree.hpp"

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace RcppEigen;

// [[Rcpp::export]]
pptree::Tree<double, int> train(
  pptree::Data<double>    data,
  pptree::DataColumn<int> groups) {
  return pptree::train_lda(data, groups);
}

// [[Rcpp::export]]
pptree::DataColumn<int> predict(
  pptree::Data<double>      data,
  pptree::Tree<double, int> tree) {
  return pptree::predict(data, tree);
}
