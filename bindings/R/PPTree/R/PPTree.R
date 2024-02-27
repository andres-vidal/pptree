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
#' PPTree(x = PPTreeIris[, 1:4], y = PPTreeIris[, 5])
#'
#' # Example 2: with the preprocessed `crabs` dataset
#' library(PPTree)
#' PPTree(x = PPTreeCrabs[, 2:7], y = PPTreeCrabs[, 1])
#'
#' @export
PPTree <- function(x, y) {
  model <- pptree_train_lda(as.matrix(x), as.matrix(y))
  class(model) <- "PPTree"
  model
}

#' Predicts the labels of a set of observations using a PPTree model.
#' 
#' @param model A PPTree model.
#' @param x A matrix containing the features for each new observation.
#' @return A matrix containing the predicted labels for each observation.
#' @examples
#' # Example 1: with the preprocessed `iris` dataset
#' library(PPTree)
#' model <- PPTree(PPTreeIris[, 1:4], PPTreeIris[, 5])
#' predict(model, PPTreeIris[, 1:4])
#'
#' # Example 2: with the preprocessed `crabs` dataset
#' library(PPTree)
#' model <- PPTree(PPTreeCrabs[, 2:7], PPTreeCrabs[, 1])
#' predict(model, PPTreeCrabs[, 2:7])
#'
#' @export
predict.PPTree <- function(model, x) {
  pptree_predict(model, as.matrix(x))
}
