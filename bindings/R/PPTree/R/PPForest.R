#' @useDynLib PPTree
#' @importFrom Rcpp evalCpp
#' @importFrom stats model.frame model.matrix model.response formula predict
NULL

#' Trains a Random Forest of Project-Pursuit oblique decision trees.
#'
#' This function trains a Random Forest of Project-Pursuit oblique decision tree (PPTree) using either a formula and data frame interface or a matrix-based interface. When using the formula interface, specify the model formula and the data frame containing the variables. For the matrix-based interface, provide matrices for the features and labels directly.
#' The number of trees is controlled by the \code{size} parameter. Each tree is trained on a stratified bootstrap sample drawn from the data.
#' The number of variables to consider at each split is controlled by the \code{n_vars} parameter.
#' If \code{lambda = 0}, the model is trained using Linear Discriminant Analysis (LDA). If \code{lambda > 0}, the model is trained using Penalized Discriminant Analysis (PDA).
#'
#' @param formula A formula of the form \code{y ~ x1 + x2 + ...}, where \code{y} is a vector of labels and \code{x1}, \code{x2}, ... are the features.
#' @param data A data frame containing the variables in the formula.
#' @param x A matrix containing the features for each observation.
#' @param y A matrix containing the labels for each observation.
#' @param size The number of trees in the forest.
#' @param lambda A regularization parameter. If \code{lambda = 0}, the model is trained using Linear Discriminant Analysis (LDA). If \code{lambda > 0}, the model is trained using Penalized Discriminant Analysis (PDA).
#' @param n_vars The number of variables to consider at each split. These are chosen uniformly in each split. The default is all variables.
#' @param n_threads The number of threads to use. The default is the number of cores available.
#' @return A PPForest model trained on \code{x} and \code{y}.
#' @examples
#'
#' # Example 1: formula interface with the `iris` dataset
#' PPForest(Type ~ ., data = iris)
#'
#' # Example 2: formula interface with the `iris` dataset with regularization
#' PPForest(Type ~ ., data = iris, lambda = 0.5)
#'
#' # Example 3: matrix interface with the `iris` dataset
#' PPForest(x = iris[, 1:4], y = iris[, 5])
#'
#' # Example 4: matrix interface with the `iris` dataset with regularization
#' PPForest(x = iris[, 1:4], y = iris[, 5], lambda = 0.5)
#'
#' # Example 5: formula interface with the `crabs` dataset
#' PPForest(Type ~ . - sex + as.numeric(as.factor(sex)), data = crabs)
#'
#' # Example 6: formula interface with the `crabs` dataset with regularization
#' PPForest(Type ~ . - sex + as.numeric(as.factor(sex)), data = crabs, lambda = 0.5)
#'
#' # Example 7: matrix interface with the `crabs` dataset
#' x <- crabs[, 2:5]
#' x$sex <- as.numeric(as.factor(crabs$sex))
#' PPForest(x = x, y = crabs$Type)
#'
#' # Example 8: matrix interface with the `crabs` dataset with regulartion
#' x <- crabs[, 2:5]
#' x$sex <- as.numeric(as.factor(crabs$sex))
#' PPForest(x = x, y = crabs$Type, lambda = 0.5)
#'
#' @export
PPForest <- function(
    formula = NULL,
    data = NULL,
    x = NULL,
    y = NULL,
    size = 2,
    lambda = 0,
    n_vars = NULL,
    n_threads = NULL) {
  args <- process_model_arguments(formula, data, x, y)

  x <- args$x
  y <- args$y
  classes <- args$classes

  model <- pptree_train_forest_glda(
    x,
    y,
    size,
    ifelse(is.null(n_vars), ncol(x), n_vars),
    lambda,
    n_threads
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
#' model <- PPForest(Type ~ ., data = iris)
#' predict(model, iris)
#'
#' # Example 2: with the `crabs` dataset
#' model <- PPForest(Type ~ . - sex + as.numeric(as.factor(sex)), data = crabs)
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
#' model <- PPForest(Type ~ ., data = iris)
#' formula(model)
#' @export
formula.PPForest <- function(x, ...) {
  x$formula
}

#' Prints a PPForest model.
#' @param x A PPForest model.
#' @param ... (unused) other parameters tipically passed to print
#' @examples
#' model <- PPForest(Type ~ ., data = iris)
#' print(model)
#'
#' @export
print.PPForest <- function(x, ...) {
  model <- x
  cat("\n")
  cat("Random Forest of Project-Pursuit Oblique Decision Tree\n")
  cat("-------------------------------------\n")
  for (i in seq_along(model$trees)) {
    cat("Tree ", i, ":\n", sep = "")
    print(model$trees[[i]])
  }
  cat("\n")
}

#' Summarizes a PPForest model.
#' @param object A PPForest model.
#' @param ... (unused) other parameters tipically passed to print
#' @examples
#' model <- PPForest(Type ~ ., data = iris)
#' summary(model)
#'
#' @export
summary.PPForest <- function(object, ...) {
  model <- object
  model$variable_importance <- data.frame(pptree_forest_variable_importance(model))
  rownames(model$variable_importance) <- colnames(model$x)
  colnames(model$variable_importance) <- c("Proj.", "Proj. Adjusted", "Permutation")

  model$confusion_matrix <- data.frame(pptree_forest_confusion_matrix(model))
  colnames(model$confusion_matrix) <- c(model$classes, "Error")
  rownames(model$confusion_matrix) <- c(model$classes, "Total")


  if (!is.null(formula(object))) {
    cat("\n")
    cat("Random Forest of Project-Pursuit Oblique Decision Tree\n")
    cat("-------------------------------------\n")
    cat("Size:", length(model$trees), "trees\n")
    cat(nrow(model$x), "observations of", ncol(model$x), "features\n")
    cat("Regularization parameter:", model$training_spec$lambda, "\n")
    cat("Classes:\n", paste(model$classes, collapse = "\n "), "\n")
    if (!is.null(model$formula)) {
      cat("Formula:\n", deparse(model$formula), "\n")
    }
    cat("-------------------------------------\n")
    cat("Variable Importance:\n")
    print(model$variable_importance)
    cat("-------------------------------------\n")
    cat("Confusion Matrix:\n")
    print(model$confusion_matrix)
  }
  cat("\n")
}
