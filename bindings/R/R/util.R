print_confusion_matrix <- function(raw_preds, model) {
  preds <- factor(model$groups[raw_preds], levels = model$groups)
  actual <- factor(model$groups[model$y], levels = model$groups)
  cm <- table(Actual = actual, Predicted = preds)
  print(cm)
  cat("\n")
  n <- length(actual)
  correct <- sum(preds == actual)
  cat("Training error: ", round((1 - correct / n) * 100, 2), "%\n", sep = "")
}

print_oob_confusion_matrix <- function(model) {
  oob_preds <- model$oob_predictions
  # 0 = sentinel for no OOB trees (was -1 in C++, +1 during conversion)
  oob_mask <- oob_preds > 0
  preds <- factor(model$groups[oob_preds[oob_mask]], levels = model$groups)
  actual <- factor(model$groups[model$y[oob_mask]], levels = model$groups)
  cm <- table(Actual = actual, Predicted = preds)
  print(cm)
  cat("\n")
  n <- length(actual)
  correct <- sum(preds == actual)
  cat("OOB error: ", round((1 - correct / n) * 100, 2), "%\n", sep = "")
}

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
    stop("`y` must have at least 2 groups.")
  }

  if (nrow(x) < nlevels(y)) {
    stop("Not enough observations: need at least as many rows as groups.")
  }

  return(
    list(
      x = x,
      y = as.matrix(as.numeric(y)),
      groups = levels(y),
      formula = formula
    )
  )
}
