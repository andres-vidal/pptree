#include <Rcpp.h>
#include <RcppEigen.h>
#include "pptree.hpp"
// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace RcppEigen;


// [[Rcpp::export]]
template<typename T, typename R>
Tree<T, R> train(
  stats::Data<T>       data,
  stats::DataColumn<R> groups,
  pp::PPStrategy<T, R> pp_strategy) {
  return pptree::train(data, groups, pp_strategy);
}

// [[Rcpp::export]]
template<typename T, typename R>
R predict(
  stats::Data<T> data,
  Tree<T, R>     tree) {
  return pptree::predict(data, tree);
}
