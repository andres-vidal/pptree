#' @useDynLib PPTree
#' @importFrom Rcpp evalCpp
#' @importFrom stats model.frame model.matrix model.response formula predict
NULL

#' Trains a Project-Pursuit oblique decision tree.
#'
#' This function trains a Project-Pursuit oblique decision tree (PPTree) using either a formula and data frame interface or a matrix-based interface. When using the formula interface, specify the model formula and the data frame containing the variables. For the matrix-based interface, provide matrices for the features and labels directly.
#' If \code{lambda = 0}, the model is trained using Linear Discriminant Analysis (LDA). If \code{lambda > 0}, the model is trained using Penalized Discriminant Analysis (PDA).
#'
#' @param formula A formula of the form \code{y ~ x1 + x2 + ...}, where \code{y} is a vector of labels and \code{x1}, \code{x2}, ... are the features.
#' @param data A data frame containing the variables in the formula.
#' @param x A matrix containing the features for each observation.
#' @param y A matrix containing the labels for each observation.
#' @param lambda A regularization parameter. If \code{lambda = 0}, the model is trained using Linear Discriminant Analysis (LDA). If \code{lambda > 0}, the model is trained using Penalized Discriminant Analysis (PDA).
#' @return A PPTree model trained on \code{x} and \code{y}.
#' @examples
#'
#' # Example 1: formula interface with the `iris` dataset
#' PPTree(Type ~ ., data = iris)
#'
#' # Example 2: formula interface with the `iris` dataset with regularization
#' PPTree(Type ~ ., data = iris, lambda = 0.5)
#'
#' # Example 3: matrix interface with the `iris` dataset
#' PPTree(x = iris[, 1:4], y = iris[, 5])
#'
#' # Example 4: matrix interface with the `iris` dataset with regularization
#' PPTree(x = iris[, 1:4], y = iris[, 5], lambda = 0.5)
#'
#' # Example 5: formula interface with the `crabs` dataset
#' PPTree(Type ~ . - sex + as.numeric(as.factor(sex)), data = crabs)
#'
#' # Example 6: formula interface with the `crabs` dataset with regularization
#' PPTree(Type ~ . - sex + as.numeric(as.factor(sex)), data = crabs, lambda = 0.5)
#'
#' # Example 7: matrix interface with the `crabs` dataset
#' x <- crabs[, 2:5]
#' x$sex <- as.numeric(as.factor(crabs$sex))
#' PPTree(x = x, y = crabs$Type)
#'
#' # Example 8: matrix interface with the `crabs` dataset with regulartion
#' x <- crabs[, 2:5]
#' x$sex <- as.numeric(as.factor(crabs$sex))
#' PPTree(x = x, y = crabs$Type, lambda = 0.5)
#'
#' @export
PPTree <- function(
    formula = NULL,
    data = NULL,
    x = NULL,
    y = NULL,
    lambda = 0) {
  args <- process_model_arguments(formula, data, x, y)

  x <- args$x
  y <- args$y
  classes <- args$classes
  formula <- args$formula

  model <- pptree_train_glda(args$x, args$y, lambda)

  class(model) <- "PPTree"
  model$classes <- classes
  model$formula <- formula
  model$x <- x
  model$y <- y

  model
}

#' Predicts the labels of a set of observations using a PPTree model.
#'
#' @param object A PPTree model.
#' @param ... (unused) other parameters tipically passed to predict.
#' @return A matrix containing the predicted labels for each observation.
#' @examples
#' # Example 1: with the `iris` dataset
#' model <- PPTree(Type ~ ., data = iris)
#' predict(model, iris)
#'
#' # Example 2: with the `crabs` dataset
#' model <- PPTree(Type ~ . - sex + as.numeric(as.factor(sex)), data = crabs)
#' predict(model, crabs)
#'
#' @export
predict.PPTree <- function(object, ...) {
  args <- list(...)
  x <- args[[1]]

  if (!is.null(object$formula)) {
    x <- model.matrix(object$formula, x)
  }

  y <- pptree_predict(object, as.matrix(x))
  as.factor(object$classes[y])
}

#' Extracts the formula used to train a PPTree model.
#'
#' @param x A PPTree model.
#' @param ... (unused) other parameters tipically passed to formula
#' @return The formula used to train the model.
#' @examples
#' model <- PPTree(Type ~ ., data = iris)
#' formula(model)
#' @export
formula.PPTree <- function(x, ...) {
  x$formula
}

print_node <- function(model, node, depth = 0) {
  indent <- paste(rep(" ", depth), collapse = "")

  if (!is.null(node$value)) {
    cat(indent, "Predict:", model$classes[node$value], "\n")
  } else {
    projection_str <- paste(
      "[", paste(round(node$projector, 2), collapse = " "), "] * x",
      collapse = ""
    )

    cat(
      indent,
      "If (", projection_str, ") < ", node$threshold, ":\n",
      sep = ""
    )

    if (!is.null(node$lower)) {
      print_node(model, node$lower, depth + 1)
    }

    cat(indent, "Else:\n", sep = "")
    if (!is.null(node$upper)) {
      print_node(model, node$upper, depth + 1)
    }
  }
}

#' Prints a PPTree model.
#' @param x A PPTree model.
#' @param ... (unused) other parameters tipically passed to print
#' @examples
#' model <- PPTree(Type ~ ., data = iris)
#' print(model)
#'
#' @export
print.PPTree <- function(x, ...) {
  model <- x
  if (!is.null(formula(x))) {
    cat("\n")
    cat("Project-Pursuit Oblique Decision Tree:\n")
  }

  print_node(model, model$root)
  cat("\n")
}

#' Summarizes a PPTree model.
#' @param object A PPTree model.
#' @param ... (unused) other parameters tipically passed to print
#' @examples
#' model <- PPTree(Type ~ ., data = iris)
#' summary(model)
#'
#' @export
summary.PPTree <- function(object, ...) {
  model <- object
  model$variable_importance <- data.frame(pptree_variable_importance(model))
  rownames(model$variable_importance) <- colnames(model$x)
  colnames(model$variable_importance) <- c("Proj.")

  model$confusion_matrix <- data.frame(pptree_confusion_matrix(model))
  colnames(model$confusion_matrix) <- c(model$classes, "Error")
  rownames(model$confusion_matrix) <- c(model$classes, "Total")


  if (!is.null(formula(object))) {
    cat("\n")
    cat("Project-Pursuit Oblique Decision Tree\n")
    cat("-------------------------------------\n")
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
