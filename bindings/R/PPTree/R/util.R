process_model_arguments <- function(formula, data, x, y) {
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

  return(
    list(
      x = as.matrix(x),
      y = as.matrix(as.numeric(y)),
      classes = levels(y)
    )
  )
}
