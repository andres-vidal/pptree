#' @useDynLib PPTree
#' @importFrom Rcpp evalCpp
#' @importFrom stats model.frame model.matrix model.response formula predict sd terms update
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
#' @param seed An optional integer seed for reproducibility. If \code{NULL} (default), a seed is drawn from R's RNG, so \code{set.seed()} controls reproducibility. If an integer is provided, that value is used directly. The same seed is used for training and for computing permuted variable importance.
#' @param n_threads The number of threads to use. The default is the number of cores available.
#' @return A PPForest model trained on \code{x} and \code{y}.
#' @seealso \code{\link{predict.PPForest}}, \code{\link{formula.PPForest}}, \code{\link{summary.PPForest}}, \code{\link{print.PPForest}}, \code{\link{pp_forest}} for parsnip integration
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
#' # Example 8: matrix interface with the `crabs` dataset with regularization
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
    seed = NULL,
    n_threads = NULL) {
  args <- process_model_arguments(formula, data, x, y)

  x <- args$x
  y <- args$y
  classes <- args$classes
  formula <- args$formula

  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1L)
  }

  model <- pptree_train_forest_glda(
    x,
    y,
    size,
    ifelse(is.null(n_vars), ncol(x), n_vars),
    lambda,
    seed,
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

  scale <- apply(x, 2, sd)
  scale[scale == 0] <- 1

  model$vi <- list(
    scale       = scale,
    projections = pptree_vi_projections_forest(model, ncol(x), scale),
    weighted    = pptree_vi_weighted(model, x, y, scale),
    permuted    = pptree_vi_permuted(model, x, y, seed)
  )
  model$oob_error <- pptree_oob_error(model, x, y)

  model
}

#' Predicts the labels or vote proportions of a set of observations using a PPForest model.
#'
#' @param object A PPForest model.
#' @param new_data A data frame or matrix of new observations to predict. If \code{NULL}, the first positional argument in \code{...} is used for backward compatibility.
#' @param type The type of prediction: \code{"class"} (default) returns a factor of predicted labels, \code{"prob"} returns a data frame of vote proportions.
#' @param ... For backward compatibility, the first positional argument is treated as \code{new_data} when \code{new_data} is \code{NULL}.
#' @return If \code{type = "class"}, a factor of predicted labels. If \code{type = "prob"}, a data frame with one column per class, where each row sums to 1.
#' @seealso \code{\link{PPForest}} for training, \code{\link{formula.PPForest}}, \code{\link{summary.PPForest}}
#' @examples
#' # Example 1: with the `iris` dataset
#' model <- PPForest(Type ~ ., data = iris)
#' predict(model, iris)
#'
#' # Example 2: with the `crabs` dataset
#' model <- PPForest(Type ~ . - sex + as.numeric(as.factor(sex)), data = crabs)
#' predict(model, crabs)
#'
#' # Example 3: vote proportions
#' model <- PPForest(Type ~ ., data = iris)
#' predict(model, iris, type = "prob")
#'
#' @export
predict.PPForest <- function(object, new_data = NULL, type = "class", ...) {
  x <- process_predict_arguments(object, new_data, ...)

  if (type == "prob") {
    probs <- pptree_predict_forest_prob(object, x)
    df <- as.data.frame(probs)
    colnames(df) <- object$classes
    return(df)
  }

  y <- pptree_predict_forest(object, x)
  as.factor(object$classes[y])
}

#' Extracts the formula used to train a PPForest model.
#'
#' @param x A PPForest model.
#' @param ... (unused) other parameters typically passed to formula.
#' @return The formula used to train the model.
#' @seealso \code{\link{PPForest}} for training, \code{\link{predict.PPForest}}, \code{\link{summary.PPForest}}
#' @examples
#' model <- PPForest(Type ~ ., data = iris)
#' formula(model)
#' @export
formula.PPForest <- function(x, ...) {
  x$formula
}

#' Prints a PPForest model.
#' @param x A PPForest model.
#' @param ... (unused) other parameters typically passed to print.
#' @seealso \code{\link{PPForest}}, \code{\link{summary.PPForest}}
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
#' @param ... (unused) other parameters typically passed to summary.
#' @seealso \code{\link{PPForest}}, \code{\link{predict.PPForest}}, \code{\link{print.PPForest}}
#' @examples
#' model <- PPForest(Type ~ ., data = iris)
#' summary(model)
#'
#' @export
summary.PPForest <- function(object, ...) {
  model <- object

  if (!is.null(model$x)) {
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
    if (model$oob_error >= 0) {
      cat("OOB error:", round(model$oob_error * 100, 2), "%\n")
    }
    cat("-------------------------------------\n")
    cat("Variable Importance:\n\n")
    p <- length(model$vi$projections)
    vnames <- if (!is.null(colnames(model$x))) colnames(model$x) else paste0("x", seq_len(p))
    ord <- order(model$vi$projections, decreasing = TRUE)
    tbl <- data.frame(
      Variable    = vnames[ord],
      sigma       = model$vi$scale[ord],
      Projection  = model$vi$projections[ord],
      Weighted    = model$vi$weighted[ord],
      Permuted    = model$vi$permuted[ord],
      row.names   = seq_len(p)
    )
    names(tbl)[2] <- "\u03c3"
    print(tbl)
    cat("-------------------------------------\n")
    cat("Confusion Matrix:\n")
    cat("TODO")
  }
  cat("\n")
}
