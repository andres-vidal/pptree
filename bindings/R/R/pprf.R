#' @useDynLib ppforest2
#' @importFrom Rcpp evalCpp
#' @importFrom stats model.frame model.matrix model.response formula predict sd terms update
NULL

#' Trains a Random Forest of Project-Pursuit oblique decision trees.
#'
#' This function trains a Random Forest of Project-Pursuit oblique decision tree using either a formula and data frame interface or a matrix-based interface. When using the formula interface, specify the model formula and the data frame containing the variables. For the matrix-based interface, provide matrices for the features and labels directly.
#' The number of trees is controlled by the \code{size} parameter. Each tree is trained on a stratified bootstrap sample drawn from the data.
#' The number of variables to consider at each split is controlled by the \code{n_vars} parameter.
#' If \code{lambda = 0}, the model is trained using Linear Discriminant Analysis (LDA). If \code{lambda > 0}, the model is trained using Penalized Discriminant Analysis (PDA).
#'
#' @param formula A formula of the form \code{y ~ x1 + x2 + ...}, where \code{y} is a vector of labels and \code{x1}, \code{x2}, ... are the features.
#' @param data A data frame containing the variables in the formula.
#' @param x A matrix containing the features for each observation.
#' @param y A matrix containing the labels for each observation.
#' @param size The number of trees in the forest.
#' @param lambda A regularization parameter. If \code{lambda = 0}, the model is trained using Linear Discriminant Analysis (LDA). If \code{lambda > 0}, the model is trained using Penalized Discriminant Analysis (PDA). Cannot be used together with \code{pp}.
#' @param n_vars The number of variables to consider at each split (integer). These are chosen uniformly in each split. The default is all variables. Cannot be used together with \code{p_vars} or \code{dr}.
#' @param p_vars The proportion of variables to consider at each split (number between 0 and 1, exclusive). For example, \code{p_vars = 0.5} uses half the features. Cannot be used together with \code{n_vars} or \code{dr}.
#' @param seed An optional integer seed for reproducibility. If \code{NULL} (default), a seed is drawn from R's RNG, so \code{set.seed()} controls reproducibility. If an integer is provided, that value is used directly. The same seed is used for training and for computing permuted variable importance.
#' @param max_retries Maximum number of retries for degenerate trees (default: 3). When a bootstrap sample yields a singular covariance matrix, the tree is retrained with a different seed up to this many times.
#' @param threads The number of threads to use. The default is the number of cores available.
#' @param pp A projection pursuit strategy object created by \code{\link{pp_pda}}. Cannot be used together with \code{lambda}.
#' @param dr A dimensionality reduction strategy object created by \code{\link{dr_uniform}} or \code{\link{dr_noop}}. Cannot be used together with \code{n_vars} or \code{p_vars}.
#' @param sr A split rule strategy object created by \code{\link{sr_mean_of_means}} (default).
#' @return A pprf model trained on \code{x} and \code{y}.
#' @seealso \code{\link{predict.pprf}}, \code{\link{formula.pprf}}, \code{\link{summary.pprf}}, \code{\link{print.pprf}}, \code{\link{save_json}}, \code{\link{load_json}}, \code{\link{pp_rand_forest}} for parsnip integration, \code{\link{pp_pda}}, \code{\link{dr_uniform}}, \code{\link{dr_noop}}, \code{\link{sr_mean_of_means}}, \code{vignette("introduction")} for a tutorial
#' @examples
#'
#' # Example 1: formula interface with the `iris` dataset
#' pprf(Type ~ ., data = iris)
#'
#' # Example 2: formula interface with the `iris` dataset with regularization
#' pprf(Type ~ ., data = iris, lambda = 0.5)
#'
#' # Example 3: matrix interface with the `iris` dataset
#' pprf(x = iris[, 1:4], y = iris[, 5])
#'
#' # Example 4: matrix interface with the `iris` dataset with regularization
#' pprf(x = iris[, 1:4], y = iris[, 5], lambda = 0.5)
#'
#' # Example 5: formula interface with the `crabs` dataset
#' pprf(Type ~ ., data = crabs)
#'
#' # Example 6: formula interface with the `crabs` dataset with regularization
#' pprf(Type ~ ., data = crabs, lambda = 0.5)
#'
#' @export
pprf <- function(
    formula = NULL,
    data = NULL,
    x = NULL,
    y = NULL,
    size = 2,
    lambda = 0,
    n_vars = NULL,
    p_vars = NULL,
    seed = NULL,
    max_retries = 3L,
    threads = NULL,
    pp = NULL,
    dr = NULL,
    sr = NULL) {
  if (!is.null(seed) && (!is.numeric(seed) || length(seed) != 1 || seed != as.integer(seed)))
    stop("`seed` must be a single integer or NULL.")

  if (!is.numeric(size) || length(size) != 1 || size < 1 || size != as.integer(size))
    stop("`size` must be a positive integer.")

  if (!is.numeric(max_retries) || length(max_retries) != 1 || max_retries < 0 || max_retries != as.integer(max_retries))
    stop("`max_retries` must be a non-negative integer.")

  if (!is.null(threads) && (!is.numeric(threads) || length(threads) != 1 || threads < 1 || threads != as.integer(threads)))
    stop("`threads` must be a positive integer or NULL.")

  args <- resolve_model_data(formula, data, x, y)

  x <- args$x
  y <- args$y
  groups <- args$groups
  formula <- args$formula

  strategies <- resolve_strategies(
    pp = pp, lambda = lambda, lambda_missing = missing(lambda),
    dr = dr, n_vars = n_vars, n_vars_missing = missing(n_vars),
    p_vars = p_vars, p_vars_missing = missing(p_vars),
    sr = sr,
    default_dr = dr_uniform(),
    n_features = ncol(x))

  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1L)
  }

  effective_threads <- if (is.null(threads)) parallel::detectCores() else threads
  if (effective_threads > 1 && !ppforest2_has_openmp()) {
    warning("OpenMP is not available. The forest will be trained using a single thread.\n",
            "On macOS, install libomp: brew install libomp", call. = FALSE)
  }

  training_spec <- list(
    pp = strategies$pp,
    dr = strategies$dr,
    sr = strategies$sr,
    size = as.integer(size),
    seed = as.integer(seed),
    threads = if (is.null(threads)) 0L else as.integer(threads),
    max_retries = as.integer(max_retries))

  model <- ppforest2_train(training_spec, x, y)

  if (isTRUE(model$degenerate)) {
    warning("Some splits could not separate groups (degenerate nodes). ",
            "This can be caused by ill-conditioned variables in the input data, ",
            "or by bootstrap samples that produce singular covariance matrices. ",
            "Consider reviewing your data or adjusting `max_retries` (currently ", max_retries, "). ",
            "Degenerate nodes predict the group with the most observations. ",
            "Degenerate trees are excluded from variable importance calculations.",
            call. = FALSE)
  }

  for (i in 1:size) {
    model$trees[[i]]$groups <- groups
  }

  model$groups <- groups
  model$formula <- formula
  model$x <- x
  model$y <- y

  scale <- apply(x, 2, sd)
  scale[scale == 0] <- 1

  model$vi <- list(
    scale       = scale,
    projections = ppforest2_vi_projections_forest(model, ncol(x), scale),
    weighted    = ppforest2_vi_weighted_forest(model, x, y, scale),
    permuted    = ppforest2_vi_permuted_forest(model, x, y, seed)
  )
  model$oob_error <- ppforest2_oob_error(model, x, y)
  model$oob_predictions <- ppforest2_oob_predict(model, x)

  model
}

#' Predicts the labels or vote proportions of a set of observations using a pprf model.
#'
#' @param object A pprf model.
#' @param new_data A data frame or matrix of new observations to predict. If \code{NULL}, the first positional argument in \code{...} is used for backward compatibility.
#' @param type The type of prediction: \code{"class"} (default) returns a factor of predicted labels, \code{"prob"} returns a data frame of vote proportions.
#' @param ... For backward compatibility, the first positional argument is treated as \code{new_data} when \code{new_data} is \code{NULL}.
#' @return If \code{type = "class"}, a factor of predicted labels. If \code{type = "prob"}, a data frame with one column per group, where each row sums to 1.
#' @seealso \code{\link{pprf}} for training, \code{\link{formula.pprf}}, \code{\link{summary.pprf}}
#' @examples
#' # Example 1: with the `iris` dataset
#' model <- pprf(Type ~ ., data = iris)
#' predict(model, iris)
#'
#' # Example 2: with the `crabs` dataset
#' model <- pprf(Type ~ ., data = crabs)
#' predict(model, crabs)
#'
#' # Example 3: vote proportions
#' model <- pprf(Type ~ ., data = iris)
#' predict(model, iris, type = "prob")
#'
#' @export
predict.pprf <- function(object, new_data = NULL, type = "class", ...) {
  x <- process_predict_arguments(object, new_data, ...)

  if (type == "prob") {
    probs <- ppforest2_predict_forest_prob(object, x)
    df <- as.data.frame(probs)
    colnames(df) <- object$groups
    return(df)
  }

  y <- ppforest2_predict_tree_forest(object, x)
  as.factor(object$groups[y])
}

#' Extracts the formula used to train a pprf model.
#'
#' @param x A pprf model.
#' @param ... (unused) other parameters typically passed to formula.
#' @return The formula used to train the model.
#' @seealso \code{\link{pprf}} for training, \code{\link{predict.pprf}}, \code{\link{summary.pprf}}
#' @examples
#' model <- pprf(Type ~ ., data = iris)
#' formula(model)
#' @export
formula.pprf <- function(x, ...) {
  x$formula
}

#' Prints a pprf model.
#' @param x A pprf model.
#' @param ... (unused) other parameters typically passed to print.
#' @seealso \code{\link{pprf}}, \code{\link{summary.pprf}}
#' @examples
#' model <- pprf(Type ~ ., data = iris)
#' print(model)
#'
#' @export
print.pprf <- function(x, ...) {
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

#' Summarizes a pprf model.
#' @param object A pprf model.
#' @param ... (unused) other parameters typically passed to summary.
#' @seealso \code{\link{pprf}}, \code{\link{predict.pprf}}, \code{\link{print.pprf}}
#' @examples
#' model <- pprf(Type ~ ., data = iris)
#' summary(model)
#'
#' @export
summary.pprf <- function(object, ...) {
  model <- object

  if (!is.null(model$x)) {
    cat("\n")
    cat("Random Forest of Project-Pursuit Oblique Decision Tree\n")
    cat("\n")
    cat("Size:", length(model$trees), "trees\n")
    print_training_spec(model$training_spec)
    cat("\n")
    cat("Data Summary:\n")
    cat("  observations:", nrow(model$x), "\n")
    cat("  features:    ", ncol(model$x), "\n")
    cat("  groups:      ", length(model$groups), "\n")
    cat("  group names: ", paste(model$groups, collapse = ", "), "\n")
    if (!is.null(model$formula)) {
      cat("  formula:     ", deparse(model$formula), "\n")
    }
    cat("\n")
    cat("Training Confusion Matrix:\n\n")
    print_confusion_matrix(ppforest2_predict_tree_forest(model, model$x), model)
    cat("\n")
    cat("OOB Confusion Matrix:\n\n")
    print_oob_confusion_matrix(model) 
    cat("\n")
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
    if (!all(model$vi$scale == 1)) {
      cat("\nNote: Variable importance was calculated using scaled coefficients (|a_j| * \u03c3_j).\n")
      cat("Variable contributions can only be theoretically interpreted as such\n")
      cat("if the model was trained on scaled data. Scaling also changes the\n")
      cat("projection-pursuit optimization, which may affect the resulting tree.\n")
    }
  }
  cat("\n")
}
