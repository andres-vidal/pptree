# Resolve strategy objects vs shortcut params into strategy objects.
#
# When shortcut params are used (lambda, n_vars, p_vars), builds the
# corresponding strategy objects. When explicit strategy objects are
# provided, validates and forwards them. Errors if both APIs are mixed.
#
# When n_features is provided, p_vars is resolved to n_vars and the
# upper bound (n_vars <= n_features) is validated for uniform DR.
#
# @param pp A pp_strategy object or NULL.
# @param lambda Shortcut for PDA lambda (caller must pass missing() result).
# @param lambda_missing TRUE if lambda was not explicitly passed by the user.
# @param dr A dr_strategy object or NULL.
# @param n_vars Shortcut for number of variables.
# @param n_vars_missing TRUE if n_vars was not explicitly passed by the user.
# @param p_vars Shortcut for proportion of variables.
# @param p_vars_missing TRUE if p_vars was not explicitly passed by the user.
# @param sr A sr_strategy object or NULL.
# @param default_dr The default DR strategy when none is specified.
# @param n_features Number of features, used to resolve p_vars and validate
#   n_vars upper bound. NULL if not yet known.
# @return A list with resolved `pp`, `dr`, and `sr` strategy objects.
resolve_strategies <- function(
  pp,
  lambda,
  lambda_missing,
  dr = NULL,
  n_vars = NULL,
  n_vars_missing = TRUE,
  p_vars = NULL,
  p_vars_missing = TRUE,
  sr = NULL,
  default_dr = dr_noop(),
  n_features = NULL) {
  if (!is.null(pp) && !lambda_missing)
    stop("Cannot use `pp` together with `lambda`. Use one API or the other.")

  if (!is.null(dr) && (!n_vars_missing || !p_vars_missing))
    stop("Cannot use `dr` together with `n_vars`/`p_vars`. Use one API or the other.")

  if (!is.null(pp) && !inherits(pp, "pp_strategy"))
    stop("`pp` must be a pp_strategy object (e.g., pp_pda()).")

  if (!is.null(dr) && !inherits(dr, "dr_strategy"))
    stop("`dr` must be a dr_strategy object (e.g., dr_uniform() or dr_noop()).")

  if (!is.null(sr) && !inherits(sr, "sr_strategy"))
    stop("`sr` must be a sr_strategy object (e.g., sr_mean_of_means()).")

  # PP strategy
  if (is.null(pp))
    pp <- pp_pda(lambda)

  # DR strategy
  if (is.null(dr)) {
    if (!is.null(n_vars) || !is.null(p_vars)) {
      dr <- dr_uniform(n_vars = n_vars, p_vars = p_vars)
    } else {
      dr <- default_dr
    }
  }

  # Resolve p_vars to n_vars now that we know the number of features
  if (!is.null(n_features) && dr$name == "uniform") {
    if (!is.null(dr$p_vars)) {
      dr$n_vars <- max(1L, as.integer(round(dr$p_vars * n_features)))
      dr$p_vars <- NULL
    }
    if (is.null(dr$n_vars)) {
      dr$n_vars <- n_features
    }
    if (dr$n_vars > n_features) {
      stop("`n_vars` must be an integer between 1 and the number of features (", n_features, ").")
    }
  }

  # SR strategy
  if (is.null(sr))
    sr <- sr_mean_of_means()

  list(pp = pp, dr = dr, sr = sr)
}

print_training_spec <- function(spec) {
  section_labels <- c(pp = "pp method", dr = "dr method", sr = "sr method")
  for (section in c("pp", "dr", "sr")) {
    s <- spec[[section]]
    if (is.null(s)) next
    display <- if (!is.null(s$display_name)) s$display_name else s$name
    params <- s[!names(s) %in% c("name", "display_name")]
    if (length(params) == 0) {
      cat(section_labels[[section]], ": ", display, "\n", sep = "")
    } else {
      param_str <- paste(names(params), params, sep = "=", collapse = ", ")
      cat(section_labels[[section]], ": ", display, " (", param_str, ")\n", sep = "")
    }
  }
  cat("\n")
}

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

resolve_model_data <- function(formula, data, x, y) {
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
