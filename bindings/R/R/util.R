process_predict_arguments <- function(object, new_data, ...) {
  if (is.null(new_data)) {
    new_data <- list(...)[[1]]
  }

  if (!is.null(object$formula)) {
    new_data <- model.matrix(object$formula, new_data)
  }

  as.matrix(new_data)
}

process_model_arguments <- function(formula, data, x, y) {
  if (!is.null(formula) && !is.null(data)) {
    if (!inherits(formula, "formula")) {
      stop("`formula` must be a formula object.")
    }

    if (!is.data.frame(data)) {
      stop("`data` must be a data frame.")
    }

    formula <- update(formula(terms(formula, data = data)), . ~ . - 1)

    y <- model.response(model.frame(formula, data))
    x <- model.matrix(formula, data, response = TRUE)
  } else if (is.null(x) || is.null(y)) {
    stop("For the matrix interface, both `x` and `y` must be provided.")
  }

  x <- as.matrix(x)

  if (!is.numeric(x)) {
    stop("All columns in `x` must be numeric.")
  }

  if (!is.factor(y)) {
    y <- factor(y)
  }

  if (nrow(x) != length(y)) {
    stop("`x` and `y` must have the same number of observations.")
  }

  if (nlevels(y) < 2) {
    stop("`y` must have at least 2 classes.")
  }

  if (nrow(x) < nlevels(y)) {
    stop("Not enough observations: need at least as many rows as classes.")
  }

  return(
    list(
      x = x,
      y = as.matrix(as.numeric(y)),
      classes = levels(y),
      formula = formula
    )
  )
}
