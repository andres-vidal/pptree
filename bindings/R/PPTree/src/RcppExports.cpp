// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "../inst/include/PPTree.h"
#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// train
pptree::Tree<long double, int> train(pptree::Data<long double> data, pptree::DataColumn<int> groups);
RcppExport SEXP _PPTree_train(SEXP dataSEXP, SEXP groupsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< pptree::Data<long double> >::type data(dataSEXP);
    Rcpp::traits::input_parameter< pptree::DataColumn<int> >::type groups(groupsSEXP);
    rcpp_result_gen = Rcpp::wrap(train(data, groups));
    return rcpp_result_gen;
END_RCPP
}
// predict
pptree::DataColumn<int> predict(pptree::Data<long double> data, pptree::Tree<long double, int> tree);
RcppExport SEXP _PPTree_predict(SEXP dataSEXP, SEXP treeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< pptree::Data<long double> >::type data(dataSEXP);
    Rcpp::traits::input_parameter< pptree::Tree<long double, int> >::type tree(treeSEXP);
    rcpp_result_gen = Rcpp::wrap(predict(data, tree));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_PPTree_train", (DL_FUNC) &_PPTree_train, 2},
    {"_PPTree_predict", (DL_FUNC) &_PPTree_predict, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_PPTree(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}