#' @useDynLib PPTree
#' @importFrom Rcpp evalCpp
#' @importFrom stats model.frame model.matrix model.response
NULL

#' Trains a Project-Pursuit oblique decision tree.
#'
#' This function trains a Project-Pursuit oblique decision tree (PPTree) using either a formula and data frame interface or a matrix-based interface. When using the formula interface, specify the model formula and the data frame containing the variables. For the matrix-based interface, provide matrices for the features and labels directly.
#'
#' @param formula A formula of the form \code{y ~ x1 + x2 + ...}, where \code{y} is a vector of labels and \code{x1}, \code{x2}, ... are the features.
#' @param data A data frame containing the variables in the formula.
#' @param x A matrix containing the features for each observation.
#' @param y A matrix containing the labels for each observation.
#' @return A PPTree model trained on \code{x} and \code{y}.
#' @examples
#' # Example 1: matrix interface with the `iris` dataset
#' PPTree(x = iris[, 1:4], y = iris[, 5])
#'
#' # Example 2: formula interface with the `iris` dataset
#' PPTree(Species ~ ., data = iris)
#'
#' # Example 3: matrix interface with the `crabs` dataset
#' x <- crabs[, c(2, 4:8)]
#' x$sex <- as.numeric(as.factor(x$sex))
#' PPTree(x = x, y = crabs[, 1])
#'
#' # Example 4: formula interface with the `crabs` dataset
#' PPTree(sp ~ . - sex + as.numeric(as.factor(sex)), data = crabs)
#'
#' @export
PPTree <- function(formula = NULL, data = NULL, x = NULL, y = NULL) {
  if (!is.null(formula) && !is.null(data)) {
    if (!inherits(formula, "formula")) {
      stop("`formula` must be a formula object.")
    }

    if (!is.data.frame(data)) {
      stop("`data` must be a data frame.")
    }

    y <- model.response(model.frame(formula, data))
    x <- model.matrix(formula, data, response = TRUE)
  } else if (is.null(x) || is.null(y)) {
    stop("For the matrix interface, both `x` and `y` must be provided.")
  }


  if (!all(sapply(x, is.numeric))) {
    stop("All columns in `x` must be numeric")
  }

  if (!is.factor(y)) {
    y <- factor(y)
  }

  model <- pptree_train_lda(as.matrix(x), as.matrix(as.numeric(y)))
  class(model) <- "PPTree"
  model$classes <- levels(y)
  model$formula <- formula
  model$x <- x
  model$y <- y

  model
}

#' Predicts the labels of a set of observations using a PPTree model.
#'
#' @param model A PPTree model.
#' @param x A matrix containing the features for each new observation.
#' @return A matrix containing the predicted labels for each observation.
#' @examples
#' # Example 1: with the `iris` dataset
#' model <- PPTree(Species ~ ., data = iris)
#' predict(model, iris)
#'
#' # Example 2: with the `crabs` dataset
#' model <- PPTree(sp ~ . - sex + as.numeric(as.factor(sex)), data = crabs)
#' predict(model, crabs)
#'
#' @export
predict.PPTree <- function(model, x) {
  if (!is.null(model$formula)) {
    x <- model.matrix(model$formula, x)
  }

  y <- pptree_predict(model, as.matrix(x))
  as.factor(model$classes[y])
}

#' Extracts the formula used to train a PPTree model.
#'
#' @param model A PPTree model.
#' @return The formula used to train the model.
#' @examples
#' model <- PPTree(Species ~ ., data = iris)
#' formula(model)
#' @export
formula.PPTree <- function(model) {
  model$formula
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
#' @param model A PPTree model.
#' @examples
#' model <- PPTree(Species ~ ., data = iris)
#' print(model)
#'
#' @export
print.PPTree <- function(model) {
  cat("\n")
  cat("Project-Pursuit Oblique Decision Tree\n")
  cat("-------------------------------------\n")
  cat(nrow(model$x), "observations of", ncol(model$x), "features\n")
  cat("Features:\n", paste(colnames(model$x), collapse = "\n "), "\n")
  cat("Classes:\n", paste(model$classes, collapse = "\n "), "\n")
  if (!is.null(model$formula)) {
    cat("Formula:\n", deparse(model$formula), "\n")
  }
  cat("-------------------------------------\n")
  cat("Structure:\n")
  print_node(model, model$root)
  cat("\n")
}
