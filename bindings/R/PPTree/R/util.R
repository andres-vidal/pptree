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


  if (!all(sapply(x, is.numeric))) {
    stop("All columns in `x` must be numeric")
  }

  if (!is.factor(y)) {
    y <- factor(y)
  }

  return(
    list(
      x = as.matrix(x),
      y = as.matrix(as.numeric(y)),
      classes = levels(y),
      formula = formula
    )
  )
}
