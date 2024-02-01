#include <Rcpp.h>
#include <RcppEigen.h>
#include "pptree.hpp"

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace RcppEigen;

// [[Rcpp::export]]
pptree::Tree<long double, int> train(
  pptree::Data<long double> data,
  pptree::DataColumn<int>   groups) {
  return pptree::train_lda(data, groups);
}

// [[Rcpp::export]]
pptree::DataColumn<int> predict(
  pptree::Data<long double>      data,
  pptree::Tree<long double, int> tree) {
  return tree.predict(data);
}
