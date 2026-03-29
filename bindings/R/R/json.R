#' Save a model to a JSON file.
#'
#' Serializes a \code{pptr} or \code{pprf} model to JSON format compatible
#' with the C++ CLI. The JSON includes the model structure, group labels,
#' training parameters, and optionally variable importance metrics.
#'
#' @param model A \code{pptr} or \code{pprf} model.
#' @param path File path to write the JSON to.
#' @param include_metrics If \code{TRUE} (default), include variable importance
#'   and (for forests) OOB error in the output. Set to \code{FALSE} to save
#'   only the model structure and metadata.
#' @param ... Additional arguments (currently unused).
#' @seealso \code{\link{load_json}}, \code{\link{pptr}}, \code{\link{pprf}}
#' @examples
#' model <- pptr(Type ~ ., data = iris, seed = 0)
#' path <- tempfile(fileext = ".json")
#' save_json(model, path)
#'
#' @export
save_json <- function(model, path, ...) {
  UseMethod("save_json")
}

#' @rdname save_json
#' @export
save_json.pptr <- function(model, path, include_metrics = TRUE, ...) {
  if (include_metrics && is.null(model$x)) {
    warning("Training data not available; saving without metrics.", call. = FALSE)
    include_metrics <- FALSE
  }
  x <- if (!is.null(model$x)) model$x else matrix(0, nrow = 0, ncol = 0)
  y <- if (!is.null(model$y)) model$y else integer(0)
  feature_names <- if (!is.null(model$x)) colnames(model$x) else character(0)
  json_str <- ppforest2_save_model_json(
    model, model$groups, include_metrics, x, y, feature_names
  )
  writeLines(json_str, path)
  invisible(path)
}

#' @rdname save_json
#' @export
save_json.pprf <- function(model, path, include_metrics = TRUE, ...) {
  if (include_metrics && is.null(model$x)) {
    warning("Training data not available; saving without metrics.", call. = FALSE)
    include_metrics <- FALSE
  }
  x <- if (!is.null(model$x)) model$x else matrix(0, nrow = 0, ncol = 0)
  y <- if (!is.null(model$y)) model$y else integer(0)
  feature_names <- if (!is.null(model$x)) colnames(model$x) else character(0)
  json_str <- ppforest2_save_model_json(
    model, model$groups, include_metrics, x, y, feature_names
  )
  writeLines(json_str, path)
  invisible(path)
}

#' Load a model from a JSON file.
#'
#' Deserializes a \code{pptr} or \code{pprf} model from a JSON file. The model
#' can be used for prediction immediately. If variable importance metrics were
#' saved, they are restored as well.
#'
#' Note that \code{formula}, \code{x}, and \code{y} are not stored in the JSON
#' and will be \code{NULL} on the loaded model. This means formula-based
#' \code{predict()} (passing a data frame) will not work; pass a numeric matrix
#' instead.
#'
#' @param path File path to read the JSON from.
#' @return A \code{pptr} or \code{pprf} model.
#' @seealso \code{\link{save_json}}, \code{\link{pptr}}, \code{\link{pprf}}
#' @examples
#' model <- pptr(Type ~ ., data = iris, seed = 0)
#' path <- tempfile(fileext = ".json")
#' save_json(model, path)
#' loaded <- load_json(path)
#' predict(loaded, as.matrix(iris[, 1:4]))
#'
#' @export
load_json <- function(path) {
  model <- ppforest2_load_model_json(path)
  model$groups <- as.character(model$groups)

  if (!is.null(model$trees)) {
    for (i in seq_along(model$trees)) {
      model$trees[[i]]$groups <- model$groups
    }
  }

  model
}
