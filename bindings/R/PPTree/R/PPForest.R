#' @useDynLib PPTree
#' @importFrom Rcpp evalCpp
#' @importFrom stats model.frame model.matrix model.response formula predict
NULL

#' Trains a Random Forest of Project-Pursuit oblique decision trees.
#'
#' This function trains a Random Forest of Project-Pursuit oblique decision tree (PPTree) using either a formula and data frame interface or a matrix-based interface. When using the formula interface, specify the model formula and the data frame containing the variables. For the matrix-based interface, provide matrices for the features and labels directly.
#' The number of trees is controlled by the \code{size} parameter. Each tree is trained on a stratified bootstrap sample drawn from the data.
#' The number of variables to consider at each split is controlled by the \code{nvars} parameter.
#' If \code{lambda = 0}, the model is trained using Linear Discriminant Analysis (LDA). If \code{lambda > 0}, the model is trained using Penalized Discriminant Analysis (PDA).
#'
#' @param formula A formula of the form \code{y ~ x1 + x2 + ...}, where \code{y} is a vector of labels and \code{x1}, \code{x2}, ... are the features.
#' @param data A data frame containing the variables in the formula.
#' @param x A matrix containing the features for each observation.
#' @param y A matrix containing the labels for each observation.
#' @param size The number of trees in the forest.
#' @param lambda A regularization parameter. If \code{lambda = 0}, the model is trained using Linear Discriminant Analysis (LDA). If \code{lambda > 0}, the model is trained using Penalized Discriminant Analysis (PDA).
#' @param nvars The number of variables to consider at each split. These are chosen uniformly in each split. The default is all variables.
#' @return A PPForest model trained on \code{x} and \code{y}.
#' @examples
#'
#' # Example 1: formula interface with the `iris` dataset
#' PPForest(Species ~ ., data = iris)
#'
#' # Example 2: formula interface with the `iris` dataset with regularization
#' PPForest(Species ~ ., data = iris, lambda = 0.5)
#'
#' # Example 3: matrix interface with the `iris` dataset
#' PPForest(x = iris[, 1:4], y = iris[, 5])
#'
#' # Example 4: matrix interface with the `iris` dataset with regularization
#' PPForest(x = iris[, 1:4], y = iris[, 5], lambda = 0.5)
#'
#' # Example 5: formula interface with the `crabs` dataset
#' PPForest(sp ~ . - sex + as.numeric(as.factor(sex)), data = crabs)
#'
#' # Example 6: formula interface with the `crabs` dataset with regularization
#' PPForest(sp ~ . - sex + as.numeric(as.factor(sex)), data = crabs, lambda = 0.5)
#'
#' # Example 7: matrix interface with the `crabs` dataset
#' x <- crabs[, c(2, 4:8)]
#' x$sex <- as.numeric(as.factor(x$sex))
#' PPForest(x = x, y = crabs[, 1])
#'
#' # Example 8: matrix interface with the `crabs` dataset with regulartion
#' x <- crabs[, c(2, 4:8)]
#' x$sex <- as.numeric(as.factor(x$sex))
#' PPForest(x = x, y = crabs[, 1], lambda = 0.5)
#'
#' @export
PPForest <- function(
    formula = NULL,
    data = NULL,
    x = NULL,
    y = NULL,
    size = 2,
    lambda = 0,
    nvars = NULL) {
  args <- process_model_arguments(formula, data, x, y)

  x <- args$x
  y <- args$y
  classes <- args$classes

  model <- pptree_train_forest_glda(
    x,
    y,
    size,
    (if (is.null(nvars)) ncol(args$x) else nvars),
    lambda
  )

  class(model) <- "PPForest"

  for (i in 1:size) {
    class(model$trees[[i]]) <- "PPTree"
    model$trees[[i]]$classes <- classes
  }

  model$classes <- classes
  model$formula <- formula
  model$x <- x
  model$y <- y

  model
}

#' Predicts the labels of a set of observations using a PPForest model.
#'
#' @param object A PPForest model.
#' @param ... other parameters tipically passed to predict.
#' @return A matrix containing the predicted labels for each observation.
#' @examples
#' # Example 1: with the `iris` dataset
#' model <- PPForest(Species ~ ., data = iris)
#' predict(model, iris)
#'
#' # Example 2: with the `crabs` dataset
#' model <- PPForest(sp ~ . - sex + as.numeric(as.factor(sex)), data = crabs)
#' predict(model, crabs)
#'
#' @export
predict.PPForest <- function(object, ...) {
  args <- list(...)
  x <- args[[1]]

  if (!is.null(object$formula)) {
    x <- model.matrix(object$formula, x)
  }

  y <- pptree_predict_forest(object, as.matrix(x))
  as.factor(object$classes[y])
}

#' Extracts the formula used to train a PPForest model.
#'
#' @param x A PPForest model.
#' @param ... (unused) other parameters tipically passed to formula
#' @return The formula used to train the model.
#' @examples
#' model <- PPForest(Species ~ ., data = iris)
#' formula(model)
#' @export
formula.PPForest <- function(x, ...) {
  x$formula
}

#' Prints a PPForest model.
#' @param x A PPForest model.
#' @param ... (unused) other parameters tipically passed to print
#' @examples
#' model <- PPForest(Species ~ ., data = iris)
#' print(model)
#'
#' @export
print.PPForest <- function(x, ...) {
  model <- x
  cat("\n")
  cat("Random Forest of Project-Pursuit Oblique Decision Tree\n")
  cat("-------------------------------------\n")
  cat("Random seed:", model$trainingSpec$seed, "\n")
  cat("Number of trees:", length(model$trees), "\n")
  cat("Number of variables in each split:", model$trainingSpec$nvars, "\n")
  cat("Lambda (Regularization Parameter):", model$trainingSpec$lambda, "\n")
  cat(nrow(model$x), "observations of", ncol(model$x), "features\n")
  cat("Features:\n", paste(colnames(model$x), collapse = "\n "), "\n")
  cat("Classes:\n", paste(model$classes, collapse = "\n "), "\n")
  if (!is.null(model$formula)) {
    cat("Formula:\n", deparse(model$formula), "\n")
  }
  cat("-------------------------------------\n")
  cat("Structure:\n")
  for (i in seq_along(model$trees)) {
    cat("Tree ", i, ":\n", sep = "")
    print(model$trees[[i]])
  }
  cat("\n")
}
