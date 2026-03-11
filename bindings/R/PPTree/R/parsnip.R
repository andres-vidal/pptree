# Suppress R CMD check NOTEs for rlang expressions used in parsnip registration
utils::globalVariables(c("object", "new_data"))

#' @keywords internal
.onLoad <- function(libname, pkgname) {
  if (requireNamespace("parsnip", quietly = TRUE)) {
    register_pp_forest()
    register_pp_tree()
  }
}

#' Parsnip model specification for PPForest.
#'
#' Creates a model specification for a Projection Pursuit random forest.
#' Use \code{set_engine("PPTree")} to select the PPTree engine.
#'
#' @param mode A character string for the model type. Only \code{"classification"} is supported.
#' @param trees The number of trees in the forest (maps to \code{size}).
#' @param mtry The number of variables to consider at each split (maps to \code{n_vars}).
#' @param penalty The regularization parameter (maps to \code{lambda}).
#' @return A parsnip model specification.
#' @seealso \code{\link{PPForest}} for the underlying training function, \code{\link{pp_tree}} for single trees
#' @examples
#' \dontrun{
#' library(parsnip)
#' spec <- pp_forest(trees = 50, mtry = 2) %>% set_engine("PPTree")
#' fit <- spec %>% fit(Type ~ ., data = iris)
#' predict(fit, iris)
#' predict(fit, iris, type = "prob")
#' }
#' @export
pp_forest <- function(mode = "classification", trees = NULL, mtry = NULL, penalty = NULL) {
  if (!requireNamespace("parsnip", quietly = TRUE)) {
    stop("Package 'parsnip' is required for pp_forest().", call. = FALSE)
  }

  args <- list(
    trees = rlang::enquo(trees),
    mtry = rlang::enquo(mtry),
    penalty = rlang::enquo(penalty)
  )

  parsnip::new_model_spec(
    "pp_forest",
    args = args,
    eng_args = NULL,
    mode = mode,
    method = NULL,
    engine = NULL
  )
}

#' Parsnip model specification for PPTree.
#'
#' Creates a model specification for a single Projection Pursuit decision tree.
#' Use \code{set_engine("PPTree")} to select the PPTree engine.
#'
#' @param mode A character string for the model type. Only \code{"classification"} is supported.
#' @param penalty The regularization parameter (maps to \code{lambda}).
#' @return A parsnip model specification.
#' @seealso \code{\link{PPTree}} for the underlying training function, \code{\link{pp_forest}} for forests
#' @examples
#' \dontrun{
#' library(parsnip)
#' spec <- pp_tree(penalty = 0) %>% set_engine("PPTree")
#' fit <- spec %>% fit(Type ~ ., data = iris)
#' predict(fit, iris)
#' }
#' @export
pp_tree <- function(mode = "classification", penalty = NULL) {
  if (!requireNamespace("parsnip", quietly = TRUE)) {
    stop("Package 'parsnip' is required for pp_tree().", call. = FALSE)
  }

  args <- list(
    penalty = rlang::enquo(penalty)
  )

  parsnip::new_model_spec(
    "pp_tree",
    args = args,
    eng_args = NULL,
    mode = mode,
    method = NULL,
    engine = NULL
  )
}

register_pp_forest <- function() {
  try(parsnip::set_new_model("pp_forest"), silent = TRUE)
  parsnip::set_model_mode("pp_forest", "classification")
  parsnip::set_model_engine("pp_forest", mode = "classification", eng = "PPTree")

  parsnip::set_encoding(
    model = "pp_forest",
    eng = "PPTree",
    mode = "classification",
    options = list(
      predictor_indicators = "none",
      compute_intercept = FALSE,
      remove_intercept = TRUE,
      allow_sparse_x = FALSE
    )
  )

  parsnip::set_model_arg(
    "pp_forest", eng = "PPTree",
    parsnip = "trees", original = "size",
    func = list(pkg = "dials", fun = "trees"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    "pp_forest", eng = "PPTree",
    parsnip = "mtry", original = "n_vars",
    func = list(pkg = "dials", fun = "mtry"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    "pp_forest", eng = "PPTree",
    parsnip = "penalty", original = "lambda",
    func = list(pkg = "dials", fun = "penalty"),
    has_submodel = FALSE
  )

  parsnip::set_fit(
    model = "pp_forest",
    eng = "PPTree",
    mode = "classification",
    value = list(
      interface = "formula",
      protect = c("formula", "data"),
      func = c(pkg = "PPTree", fun = "PPForest"),
      defaults = list()
    )
  )

  parsnip::set_pred(
    model = "pp_forest",
    eng = "PPTree",
    mode = "classification",
    type = "class",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        new_data = rlang::expr(new_data),
        type = "class"
      )
    )
  )

  parsnip::set_pred(
    model = "pp_forest",
    eng = "PPTree",
    mode = "classification",
    type = "prob",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        new_data = rlang::expr(new_data),
        type = "prob"
      )
    )
  )
}

register_pp_tree <- function() {
  try(parsnip::set_new_model("pp_tree"), silent = TRUE)
  parsnip::set_model_mode("pp_tree", "classification")
  parsnip::set_model_engine("pp_tree", mode = "classification", eng = "PPTree")

  parsnip::set_encoding(
    model = "pp_tree",
    eng = "PPTree",
    mode = "classification",
    options = list(
      predictor_indicators = "none",
      compute_intercept = FALSE,
      remove_intercept = TRUE,
      allow_sparse_x = FALSE
    )
  )

  parsnip::set_model_arg(
    "pp_tree", eng = "PPTree",
    parsnip = "penalty", original = "lambda",
    func = list(pkg = "dials", fun = "penalty"),
    has_submodel = FALSE
  )

  parsnip::set_fit(
    model = "pp_tree",
    eng = "PPTree",
    mode = "classification",
    value = list(
      interface = "formula",
      protect = c("formula", "data"),
      func = c(pkg = "PPTree", fun = "PPTree"),
      defaults = list()
    )
  )

  parsnip::set_pred(
    model = "pp_tree",
    eng = "PPTree",
    mode = "classification",
    type = "class",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        new_data = rlang::expr(new_data),
        type = "class"
      )
    )
  )

  parsnip::set_pred(
    model = "pp_tree",
    eng = "PPTree",
    mode = "classification",
    type = "prob",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        new_data = rlang::expr(new_data),
        type = "prob"
      )
    )
  )
}
