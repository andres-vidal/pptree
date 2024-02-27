#' @useDynLib PPTree
#' @importFrom Rcpp evalCpp
NULL

#' Trains a Project-Pursuit oblique decision tree.
#'
#' @param x A matrix containing the features for each observation.
#' @param y A matrix containing the labels for each observation.
#' @return A PPTree model trained on \code{x} and \code{y}.
#' @examples
#' # Example 1: with the preprocessed `iris` dataset
#' library(PPTree)
#' x <- as.matrix(PPTreeIris[, 1:4])
#' y <- as.matrix(PPTreeIris[, 5])
#' PPTree(x, y)
#'
#' # Example 2: with the preprocessed `crabs` dataset
#' library(PPTree)
#' x <- as.matrix(PPTreeCrabs[, 2:7])
#' y <- as.matrix(PPTreeCrabs[, 1])
#' PPTree(x, y)
#'
#' @export
PPTree <- function(x, y) {
  tree <- pptree_train_lda(x, y)
  class(tree) <- "PPTree"
  tree
}

#' Predicts the labels of a set of observations using a PPTree model.
#' @param model A PPTree model.
#' @param x A matrix containing the features for each new observation.
#' @return A matrix containing the predicted labels for each observation.
#' @examples
#' # Example 1: with the preprocessed `iris` dataset
#' library(PPTree)
#' x <- as.matrix(PPTreeIris[, 1:4])
#' y <- as.matrix(PPTreeIris[, 5])
#' model <- PPTree(x, y)
#' predict(model, x)
#'
#' # Example 2: with the preprocessed `crabs` dataset
#' library(PPTree)
#' x <- as.matrix(PPTreeCrabs[, 2:7])
#' y <- as.matrix(PPTreeCrabs[, 1])
#' model <- PPTree(x, y)
#' predict(model, x)
#'
#' @export
predict.PPTree <- function(model, x) {
  pptree_predict(model, x)
}
