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

// pptree_train_glda
Tree<long double, int> pptree_train_glda(const Data<long double>& data, const DataColumn<int>& groups, const double lambda, const int max_retries);
RcppExport SEXP _PPTree_pptree_train_glda(SEXP dataSEXP, SEXP groupsSEXP, SEXP lambdaSEXP, SEXP max_retriesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Data<long double>& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const DataColumn<int>& >::type groups(groupsSEXP);
    Rcpp::traits::input_parameter< const double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const int >::type max_retries(max_retriesSEXP);
    rcpp_result_gen = Rcpp::wrap(pptree_train_glda(data, groups, lambda, max_retries));
    return rcpp_result_gen;
END_RCPP
}
// pptree_train_forest_glda
Forest<long double, int> pptree_train_forest_glda(const Data<long double>& data, const DataColumn<int>& groups, const int size, const int n_vars, const double lambda, const int max_retries, SEXP n_threads);
RcppExport SEXP _PPTree_pptree_train_forest_glda(SEXP dataSEXP, SEXP groupsSEXP, SEXP sizeSEXP, SEXP n_varsSEXP, SEXP lambdaSEXP, SEXP max_retriesSEXP, SEXP n_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Data<long double>& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const DataColumn<int>& >::type groups(groupsSEXP);
    Rcpp::traits::input_parameter< const int >::type size(sizeSEXP);
    Rcpp::traits::input_parameter< const int >::type n_vars(n_varsSEXP);
    Rcpp::traits::input_parameter< const double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const int >::type max_retries(max_retriesSEXP);
    Rcpp::traits::input_parameter< SEXP >::type n_threads(n_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(pptree_train_forest_glda(data, groups, size, n_vars, lambda, max_retries, n_threads));
    return rcpp_result_gen;
END_RCPP
}
// pptree_predict
DataColumn<int> pptree_predict(const Tree<long double, int>& tree, const Data<long double>& data);
RcppExport SEXP _PPTree_pptree_predict(SEXP treeSEXP, SEXP dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Tree<long double, int>& >::type tree(treeSEXP);
    Rcpp::traits::input_parameter< const Data<long double>& >::type data(dataSEXP);
    rcpp_result_gen = Rcpp::wrap(pptree_predict(tree, data));
    return rcpp_result_gen;
END_RCPP
}
// pptree_predict_forest
DataColumn<int> pptree_predict_forest(const Forest<long double, int>& forest, const Data<long double>& data);
RcppExport SEXP _PPTree_pptree_predict_forest(SEXP forestSEXP, SEXP dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Forest<long double, int>& >::type forest(forestSEXP);
    Rcpp::traits::input_parameter< const Data<long double>& >::type data(dataSEXP);
    rcpp_result_gen = Rcpp::wrap(pptree_predict_forest(forest, data));
    return rcpp_result_gen;
END_RCPP
}
// pptree_variable_importance
Data<long double> pptree_variable_importance(const Tree<long double, int>& tree);
RcppExport SEXP _PPTree_pptree_variable_importance(SEXP treeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Tree<long double, int>& >::type tree(treeSEXP);
    rcpp_result_gen = Rcpp::wrap(pptree_variable_importance(tree));
    return rcpp_result_gen;
END_RCPP
}
// pptree_forest_variable_importance
Data<long double> pptree_forest_variable_importance(const Forest<long double, int>& forest);
RcppExport SEXP _PPTree_pptree_forest_variable_importance(SEXP forestSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Forest<long double, int>& >::type forest(forestSEXP);
    rcpp_result_gen = Rcpp::wrap(pptree_forest_variable_importance(forest));
    return rcpp_result_gen;
END_RCPP
}
// pptree_confusion_matrix
Data<long double> pptree_confusion_matrix(const Tree<long double, int>& tree);
RcppExport SEXP _PPTree_pptree_confusion_matrix(SEXP treeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Tree<long double, int>& >::type tree(treeSEXP);
    rcpp_result_gen = Rcpp::wrap(pptree_confusion_matrix(tree));
    return rcpp_result_gen;
END_RCPP
}
// pptree_forest_confusion_matrix
Data<long double> pptree_forest_confusion_matrix(const Forest<long double, int>& forest);
RcppExport SEXP _PPTree_pptree_forest_confusion_matrix(SEXP forestSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Forest<long double, int>& >::type forest(forestSEXP);
    rcpp_result_gen = Rcpp::wrap(pptree_forest_confusion_matrix(forest));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_PPTree_pptree_train_glda", (DL_FUNC) &_PPTree_pptree_train_glda, 4},
    {"_PPTree_pptree_train_forest_glda", (DL_FUNC) &_PPTree_pptree_train_forest_glda, 7},
    {"_PPTree_pptree_predict", (DL_FUNC) &_PPTree_pptree_predict, 2},
    {"_PPTree_pptree_predict_forest", (DL_FUNC) &_PPTree_pptree_predict_forest, 2},
    {"_PPTree_pptree_variable_importance", (DL_FUNC) &_PPTree_pptree_variable_importance, 1},
    {"_PPTree_pptree_forest_variable_importance", (DL_FUNC) &_PPTree_pptree_forest_variable_importance, 1},
    {"_PPTree_pptree_confusion_matrix", (DL_FUNC) &_PPTree_pptree_confusion_matrix, 1},
    {"_PPTree_pptree_forest_confusion_matrix", (DL_FUNC) &_PPTree_pptree_forest_confusion_matrix, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_PPTree(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
