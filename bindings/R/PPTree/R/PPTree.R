#' @useDynLib PPTree
#' @importFrom Rcpp evalCpp
NULL

#' Trains a Project-Pursuit oblique decision tree.
#'
#' @param x A matrix containing the features for each observation.
#' @param y A matrix containing the labels for each observation.
#' @return A PPTree model trained on \code{x} and \code{y}.
#' @examples
#' # Example 1: with the `iris` dataset
#' PPTree(x = iris[, 1:4], y = iris[, 5])
#'
#' # Example 2: with the `crabs` dataset
#' x <- crabs[, c(2, 4:8)]
#' x$sex <- as.numeric(as.factor(x$sex))
#' PPTree(x = x, y = crabs[, 1])
#'
#' @export
PPTree <- function(x, y) {
  if (!all(sapply(x, is.numeric))) {
    stop("All columns in x must be numeric")
  }

  if (!is.factor(y)) {
    y <- factor(y)
  }
  model <- pptree_train_lda(as.matrix(x), as.matrix(as.numeric(y)))
  class(model) <- "PPTree"
  model$classes <- levels(y)

  model
}

#' Predicts the labels of a set of observations using a PPTree model.
#'
#' @param model A PPTree model.
#' @param x A matrix containing the features for each new observation.
#' @return A matrix containing the predicted labels for each observation.
#' @examples
#' # Example 1: with the `iris` dataset
#' model <- PPTree(iris[, 1:4], iris[, 5])
#' predict(model, iris[, 1:4])
#'
#' # Example 2: with the `crabs` dataset
#' x <- crabs[, c(2, 4:8)]
#' x$sex <- as.numeric(as.factor(x$sex))
#' model <- PPTree(x = x, y = crabs[, 1])
#' predict(model, x)
#'
#' @export
predict.PPTree <- function(model, x) {
  y <- pptree_predict(model, as.matrix(x))
  as.factor(model$classes[y])
}
