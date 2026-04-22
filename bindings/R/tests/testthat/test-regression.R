# Regression support — tree, forest, save/load, strategies

test_that("pprf regression end-to-end on mtcars (formula + predict + summary)", {
  # Real-dataset round-trip counterpart to the simulated-data tests above.
  # mtcars is small (32 x 11) and ships with base R so this doesn't grow
  # the package footprint, while still exercising the full pipeline on
  # real covariances: formula interface, regression auto-detection from
  # numeric `y`, training, prediction, and summary rendering.
  data(mtcars)
  model <- pprf(mpg ~ ., data = mtcars, size = 10, seed = 0, threads = 1)

  expect_equal(model$mode, "regression")
  # `groups` is an empty character vector for regression (no class levels).
  expect_length(model$groups, 0)
  expect_s3_class(model, "pprf_regression")
  expect_s3_class(model, "pprf")
  expect_s3_class(model, "ppmodel")

  preds <- predict(model, newdata = mtcars)
  expect_true(is.numeric(preds))
  expect_length(preds, nrow(mtcars))
  # Predictions should land in the mpg range (no NAs, finite).
  expect_true(all(is.finite(preds)))

  # summary() must render the regression-shaped metrics block (MSE /
  # MAE / R²) and must not render a confusion matrix.
  out <- capture.output(summary(model))
  expect_true(any(grepl("Training", out)))
  expect_true(any(grepl("MSE|R\\^2|R\u00b2|MAE", out)))
  expect_false(any(grepl("Confusion Matrix", out)))

  # Sanity on OOB error: MSE is non-negative, or NA if no row has OOB.
  err <- oob_error(model)
  expect_true(is.na(err) || err >= 0)
})


make_regression_data <- function(n = 30, seed = 0) {
  set.seed(seed)
  x <- matrix(runif(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- x[, 1] + x[, 2] + rnorm(n, sd = 0.1)
  list(x = x, y = y)
}

test_that("pptr auto-detects regression from numeric y", {
  d <- make_regression_data()
  model <- pptr(x = d$x, y = d$y, seed = 0)

  expect_equal(model$mode, "regression")
  expect_length(model$groups, 0)
  # Unified `y`: continuous response stored on `model$y` for both modes.
  expect_equal(length(model$y), 30)
})

test_that("pptr classification still works with factor y", {
  data(iris)
  model <- pptr(x = iris[, 1:4], y = iris$Species, seed = 0)

  expect_equal(model$mode, "classification")
  expect_equal(length(model$groups), 3)
})

test_that("predict.pptr returns numeric vector for regression", {
  d <- make_regression_data()
  model <- pptr(x = d$x, y = d$y, seed = 0)
  preds <- predict(model, d$x)

  expect_type(preds, "double")
  expect_length(preds, 30)
  expect_false(is.factor(preds))
})

test_that("predict.pptr rejects type='class' for regression", {
  d <- make_regression_data()
  model <- pptr(x = d$x, y = d$y, seed = 0)

  expect_error(predict(model, d$x, type = "class"), "regression")
  expect_error(predict(model, d$x, type = "prob"), "regression")
})

test_that("pprf auto-detects regression from numeric y", {
  d <- make_regression_data(n = 40)
  model <- pprf(x = d$x, y = d$y, size = 5, seed = 0, threads = 1)

  expect_equal(model$mode, "regression")
  expect_length(model$trees, 5)
})

test_that("predict.pprf returns numeric for regression", {
  d <- make_regression_data(n = 40)
  model <- pprf(x = d$x, y = d$y, size = 5, seed = 0, threads = 1)
  preds <- predict(model, d$x)

  expect_type(preds, "double")
  expect_length(preds, 40)
})

test_that("pprf regression produces reasonable MSE", {
  d <- make_regression_data(n = 50)
  model <- pprf(x = d$x, y = d$y, size = 20, seed = 0, threads = 1)
  preds <- predict(model, d$x)
  mse <- mean((preds - d$y)^2)

  expect_lt(mse, 1.0)
})

test_that("pprf regression OOB error is MSE", {
  d <- make_regression_data(n = 40)
  model <- pprf(x = d$x, y = d$y, size = 10, seed = 0, threads = 1)

  expect_true(is.numeric(oob_error(model)))
  expect_gte(oob_error(model), 0)
})

test_that("pprf regression oob_predictions uses NA_real_ (not NaN) for no-OOB rows", {
  # C++ emits NaN for rows that no tree left out-of-bag; the R boundary
  # remaps those to NA_real_ so the missing-value story is uniform with
  # the classification path (which returns NA at the factor level). This
  # test pins that remap: force every row to be in-bag in every tree by
  # rewriting `sample_indices` to cover `0:(n-1)`, then confirm
  #   (a) every entry is NA under `is.na()`, and
  #   (b) no entry is the raw C++ `NaN` sentinel.
  # If someone removes the remap, (b) fails loudly instead of quietly
  # leaking a NaN that `is.na()` would still catch.
  d <- make_regression_data(n = 20)
  model <- pprf(x = d$x, y = d$y, size = 3, seed = 0, threads = 1)
  n <- nrow(model$x)
  all_in_bag <- seq_len(n) - 1L
  for (i in seq_along(model$trees)) {
    model$trees[[i]]$sample_indices <- all_in_bag
  }
  if (!is.null(model$.cache) && exists("oob_predictions", envir = model$.cache, inherits = FALSE)) {
    rm("oob_predictions", envir = model$.cache)
  }
  preds <- oob_predictions(model)
  expect_true(all(is.na(preds)))
  expect_false(any(is.nan(preds)))
})

test_that("pprf regression variable importance uses the continuous response", {
  # Regression VI (both permuted and weighted) needs the continuous response
  # as the OOB truth. Construct a dataset where x1 and x2 drive y and x3
  # is pure noise; assert the informative columns outrank the noise column
  # for both measures.
  set.seed(7)
  n <- 100
  x <- matrix(runif(n * 3), ncol = 3)
  colnames(x) <- c("x1", "x2", "noise")
  y <- 2 * x[, 1] + 3 * x[, 2] + rnorm(n, sd = 0.1)

  model <- pprf(x = x, y = y, size = 20, seed = 0, threads = 1)
  expect_equal(model$mode, "regression")

  vi_perm     <- permuted_importance(model)
  vi_weighted <- weighted_importance(model)

  expect_length(vi_perm, 3)
  expect_length(vi_weighted, 3)
  expect_true(all(is.finite(vi_perm)))
  expect_true(all(is.finite(vi_weighted)))

  # Informative columns must outrank the noise column. (Not asserting the
  # ordering between x1 and x2 — coefficient-to-VI isn't monotonic for small
  # forests, and flakes across RNG seeds.)
  expect_gt(vi_perm[1], vi_perm[3])
  expect_gt(vi_perm[2], vi_perm[3])
  expect_gt(vi_weighted[1], vi_weighted[3])
  expect_gt(vi_weighted[2], vi_weighted[3])
})

test_that("pprf regression VI is reproducible across runs", {
  set.seed(11)
  n <- 60
  x <- matrix(runif(n * 2), ncol = 2)
  y <- x[, 1] - x[, 2] + rnorm(n, sd = 0.05)

  m1 <- pprf(x = x, y = y, size = 10, seed = 0, threads = 1)
  m2 <- pprf(x = x, y = y, size = 10, seed = 0, threads = 1)

  expect_equal(permuted_importance(m1), permuted_importance(m2))
  expect_equal(weighted_importance(m1), weighted_importance(m2))
})

test_that("regression supports formula interface", {
  d <- make_regression_data()
  df <- data.frame(x1 = d$x[, 1], x2 = d$x[, 2], y = d$y)
  model <- pptr(y ~ ., data = df, seed = 0)

  expect_equal(model$mode, "regression")
  preds <- predict(model, df)
  expect_type(preds, "double")
})

test_that("regression save/load round-trip preserves predictions", {
  d <- make_regression_data()
  model <- pptr(x = d$x, y = d$y, seed = 0)

  path <- tempfile(fileext = ".json")
  save_json(model, path)
  loaded <- load_json(path)

  expect_equal(loaded$mode, "regression")

  original_preds <- predict(model, d$x)
  loaded_preds <- predict(loaded, d$x)

  expect_equal(as.numeric(loaded_preds), as.numeric(original_preds))
})

test_that("regression forest save/load round-trip preserves predictions", {
  d <- make_regression_data(n = 40)
  model <- pprf(x = d$x, y = d$y, size = 5, seed = 0, threads = 1)

  path <- tempfile(fileext = ".json")
  save_json(model, path)
  loaded <- load_json(path)

  expect_equal(loaded$mode, "regression")

  original_preds <- predict(model, d$x)
  loaded_preds <- predict(loaded, d$x)

  expect_equal(as.numeric(loaded_preds), as.numeric(original_preds))
})

test_that("stop_min_size creates valid strategy", {
  s <- stop_min_size(5L)
  expect_s3_class(s, "stop_strategy")
  expect_equal(s$name, "min_size")
  expect_equal(s$min_size, 5L)

  expect_error(stop_min_size(0L), ">= 2")
  expect_error(stop_min_size(1L), ">= 2")
})

test_that("stop_min_variance creates valid strategy", {
  s <- stop_min_variance(0.01)
  expect_s3_class(s, "stop_strategy")
  expect_equal(s$name, "min_variance")
  expect_equal(s$threshold, 0.01)

  expect_error(stop_min_variance(-1), "non-negative")
})

test_that("stop_any combines rules", {
  s <- stop_any(stop_min_size(5L), stop_min_variance(0.01))
  expect_s3_class(s, "stop_strategy")
  expect_equal(s$name, "any")
  expect_length(s$rules, 2L)

  expect_error(stop_any(), "at least one stop rule")
  expect_error(stop_any("not a rule"), "stop_strategy")
})

test_that("leaf_mean_response creates valid strategy", {
  s <- leaf_mean_response()
  expect_s3_class(s, "leaf_strategy")
  expect_equal(s$name, "mean_response")
})

test_that("grouping_by_cutpoint creates valid strategy", {
  s <- grouping_by_cutpoint()
  expect_s3_class(s, "grouping_strategy")
  expect_equal(s$name, "by_cutpoint")
})

test_that("regression with custom strategies works", {
  d <- make_regression_data(n = 50)
  model <- pptr(
    x = d$x, y = d$y, seed = 0,
    stop = stop_min_size(10),
    leaf = leaf_mean_response(),
    grouping = grouping_by_cutpoint()
  )

  expect_equal(model$mode, "regression")
  preds <- predict(model, d$x)
  expect_length(preds, 50)
})

test_that("regression is reproducible with same seed", {
  d <- make_regression_data()

  m1 <- pptr(x = d$x, y = d$y, seed = 42)
  m2 <- pptr(x = d$x, y = d$y, seed = 42)

  expect_equal(as.numeric(predict(m1, d$x)), as.numeric(predict(m2, d$x)))
})

# Non-finite y rejection — the values used to cross the C++ boundary and
# either trip an out-of-range cast (classification with NA) or violate
# strict-weak-order in std::stable_sort (regression with NaN). Rejecting
# in `validate_data` fails fast with an actionable message.
test_that("regression rejects NA in y", {
  d <- make_regression_data()
  d$y[3] <- NA_real_
  expect_error(pptr(x = d$x, y = d$y), "`y` must not contain NA")
})

test_that("regression rejects NaN in y", {
  d <- make_regression_data()
  d$y[5] <- NaN
  expect_error(pptr(x = d$x, y = d$y), "`y` must not contain NA")
})

test_that("regression rejects Inf in y", {
  d <- make_regression_data()
  d$y[7] <- Inf
  expect_error(pptr(x = d$x, y = d$y), "finite values for regression")
})

test_that("classification rejects NA in factor y", {
  x <- as.matrix(iris[, 1:4])
  y <- iris$Species
  y[10] <- NA
  expect_error(pptr(x = x, y = y), "`y` must not contain NA")
})
