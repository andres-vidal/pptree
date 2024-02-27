#include <Rcpp.h>
#include <RcppEigen.h>
#include "pptree.hpp"

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace RcppEigen;

// [[Rcpp::export]]
pptree::Tree<long double, int> pptree_train_lda(
  pptree::Data<long double> data,
  pptree::DataColumn<int>   groups) {
  return pptree::train_lda(data, groups);
}

// [[Rcpp::export]]
pptree::DataColumn<int> pptree_predict(
  pptree::Tree<long double, int> tree,
  pptree::Data<long double>      data) {
  return tree.predict(data);
}
