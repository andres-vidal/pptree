Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

describe("pptr formula interface", {
  it("returns a object with s3 class pptr", {
    model <- pptr(Species ~ ., data = iris)
    expect_s3_class(model, "pptr")
  })

  it("preserves the observations (x) as a matrix in the returned model", {
    model <- pptr(Species ~ ., data = iris)
    expected <- model.matrix(
      Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 1,
      iris,
      response = TRUE
    )
    expect_equal(model$x, expected)
  })

  it("preserves the labels (y) as a matrix in the returned model", {
    model <- pptr(Species ~ ., data = iris)
    expected <- as.matrix(as.numeric(as.factor(iris$Species)))
    expect_equal(model$y, expected)
  })

  it("preserves the groups in the returned model", {
    model <- pptr(Species ~ ., data = iris)
    expected <- levels(iris$Species)
    expect_equal(model$groups, expected)
  })

  it("preserves the formula in the returned model", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(
      model$formula,
      Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 1
    )
  })
})

describe("pptr matrix interface", {
  it("returns a object with s3 class pptr", {
    model <- pptr(x = iris[, 1:4], y = iris[, 5])
    expect_s3_class(model, "pptr")
  })

  it("preserves the observations (x) as a matrix in the returned model", {
    model <- pptr(x = iris[, 1:4], y = iris[, 5])
    expected <- as.matrix(iris[, 1:4])
    expect_equal(model$x, expected)
  })

  it("preserves the labels (y) as a matrix in the returned model", {
    model <- pptr(x = iris[, 1:4], y = iris[, 5])
    expected <- as.matrix(as.numeric(as.factor(iris$Species)))
    expect_equal(model$y, expected)
  })

  it("preserves the groups in the returned model", {
    model <- pptr(x = iris[, 1:4], y = iris[, 5])
    expected <- levels(iris$Species)
    expect_equal(model$groups, expected)
  })

  it("does not have a formula", {
    model <- pptr(x = iris[, 1:4], y = iris[, 5])
    expect_equal(model$formula, NULL)
  })
})

describe("pptr reproducibility", {
  it("produces identical predictions with set.seed", {
    set.seed(0)
    model1 <- pptr(Species ~ ., data = iris)
    set.seed(0)
    model2 <- pptr(Species ~ ., data = iris)
    expect_equal(predict(model1, iris), predict(model2, iris))
  })

  it("produces identical predictions with explicit seed", {
    model1 <- pptr(Species ~ ., data = iris, seed = 123L)
    model2 <- pptr(Species ~ ., data = iris, seed = 123L)
    expect_equal(predict(model1, iris), predict(model2, iris))
  })

  it("stores the explicit seed in the model", {
    model <- pptr(Species ~ ., data = iris, seed = 0)
    expect_equal(model$seed, 0)
  })

  it("stores the generated seed when seed is NULL", {
    set.seed(99)
    model <- pptr(Species ~ ., data = iris)
    expect_true(is.numeric(model$seed))
    expect_true(model$seed > 0)
  })
})

describe("pptr input validation", {
  it("rejects invalid lambda", {
    expect_error(pptr(x = iris[, 1:4], y = iris[, 5], lambda = -1), "between 0 and 1")
    expect_error(pptr(x = iris[, 1:4], y = iris[, 5], lambda = 2), "between 0 and 1")
    expect_error(pptr(x = iris[, 1:4], y = iris[, 5], lambda = "a"), "between 0 and 1")
  })

  it("rejects non-integer seed", {
    expect_error(pptr(x = iris[, 1:4], y = iris[, 5], seed = 1.5), "integer")
  })

  it("rejects single-group y", {
    expect_error(pptr(x = iris[, 1:4], y = rep("a", 150)), "at least 2 groups")
  })

  it("rejects dimension mismatch", {
    expect_error(pptr(x = iris[1:10, 1:4], y = iris[, 5]), "same number of observations")
  })
})

describe("pptr training spec", {
  it("preserves the lambda parameter in the returned model", {
    model <- pptr(Species ~ ., data = iris, lambda = 0.5)
    expect_equal(model$training_spec$pp$lambda, 0.5)
  })

  it("the lambda parameter is 0 by default", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(model$training_spec$pp$lambda, 0)
  })

  it("the pp strategy is pda", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(model$training_spec$pp$name, "pda")
  })

  it("the vars strategy is all", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(model$training_spec$vars$name, "all")
  })

  it("the stop strategy is pure_node", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(model$training_spec$stop$name, "pure_node")
  })

  it("the binarize strategy is largest_gap", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(model$training_spec$binarize$name, "largest_gap")
  })

  it("the grouping strategy is by_label", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(model$training_spec$grouping$name, "by_label")
  })

  it("the leaf strategy is majority_vote", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(model$training_spec$leaf$name, "majority_vote")
  })
})

describe("pptr with strategy objects", {
  it("trains with pp = pp_pda()", {
    model <- pptr(Species ~ ., data = iris, pp = pp_pda(0.5), seed = 0)
    expect_s3_class(model, "pptr")
    expect_equal(model$training_spec$pp$lambda, 0.5)
  })

  it("trains with pp = pp_pda(0) (LDA)", {
    model <- pptr(Species ~ ., data = iris, pp = pp_pda(0), seed = 0)
    expect_s3_class(model, "pptr")
    expect_equal(model$training_spec$pp$lambda, 0)
  })

  it("produces identical export as lambda shortcut", {
    model_shortcut <- pptr(Species ~ ., data = iris, lambda = 0.5, seed = 0)
    model_strategy <- pptr(Species ~ ., data = iris, pp = pp_pda(0.5), seed = 0)

    path_shortcut <- tempfile(fileext = ".json")
    path_strategy <- tempfile(fileext = ".json")
    save_json(model_shortcut, path_shortcut)
    save_json(model_strategy, path_strategy)

    j_shortcut <- jsonlite::fromJSON(readLines(path_shortcut, warn = FALSE))
    j_strategy <- jsonlite::fromJSON(readLines(path_strategy, warn = FALSE))

    # Compare full export (model, config, meta, metrics)
    j_shortcut$training_duration_ms <- NULL
    j_strategy$training_duration_ms <- NULL
    expect_equal(j_shortcut, j_strategy)
  })

  it("trains with explicit stop, binarize, grouping, leaf", {
    model <- pptr(Species ~ ., data = iris, stop = stop_pure_node(), binarize = binarize_largest_gap(), grouping = grouping_by_label(), leaf = leaf_majority_vote(), seed = 0)
    expect_s3_class(model, "pptr")
    expect_equal(model$training_spec$stop$name, "pure_node")
    expect_equal(model$training_spec$binarize$name, "largest_gap")
    expect_equal(model$training_spec$grouping$name, "by_label")
    expect_equal(model$training_spec$leaf$name, "majority_vote")
  })

  it("rejects non-stop_strategy objects", {
    expect_error(
      pptr(Species ~ ., data = iris, stop = list(name = "pure_node")),
      "stop_strategy")
  })

  it("rejects non-binarize_strategy objects", {
    expect_error(
      pptr(Species ~ ., data = iris, binarize = list(name = "largest_gap")),
      "binarize_strategy")
  })

  it("rejects non-grouping_strategy objects", {
    expect_error(
      pptr(Species ~ ., data = iris, grouping = list(name = "by_label")),
      "grouping_strategy")
  })

  it("rejects non-leaf_strategy objects", {
    expect_error(
      pptr(Species ~ ., data = iris, leaf = list(name = "majority_vote")),
      "leaf_strategy")
  })

  it("default tree and fully explicit defaults produce identical export", {
    model_default  <- pptr(Species ~ ., data = iris, seed = 0)
    model_explicit <- pptr(Species ~ ., data = iris, pp = pp_pda(0), cutpoint = cutpoint_mean_of_means(), stop = stop_pure_node(), binarize = binarize_largest_gap(), grouping = grouping_by_label(), leaf = leaf_majority_vote(), seed = 0)

    path_default  <- tempfile(fileext = ".json")
    path_explicit <- tempfile(fileext = ".json")
    save_json(model_default, path_default)
    save_json(model_explicit, path_explicit)

    j_default  <- jsonlite::fromJSON(readLines(path_default, warn = FALSE))
    j_explicit <- jsonlite::fromJSON(readLines(path_explicit, warn = FALSE))

    j_default$training_duration_ms  <- NULL
    j_explicit$training_duration_ms <- NULL
    expect_equal(j_default, j_explicit)
  })

  it("errors when mixing pp and lambda", {
    expect_error(
      pptr(Species ~ ., data = iris, pp = pp_pda(0.5), lambda = 0.3),
      "Cannot use `pp` together with `lambda`")
  })

  it("rejects non-pp_strategy objects", {
    expect_error(
      pptr(Species ~ ., data = iris, pp = list(name = "pda", lambda = 0.5)),
      "pp_strategy")
  })

  it("rejects non-cutpoint_strategy objects", {
    expect_error(
      pptr(Species ~ ., data = iris, cutpoint = list(name = "mean_of_means")),
      "cutpoint_strategy")
  })
})

describe("pptr edge cases", {
  it("rejects NA in features via matrix interface", {
    x <- matrix(c(1, NA, 3, 4, 1, 2, 3, 4), ncol = 2)
    y <- c("a", "a", "b", "b")
    expect_error(pptr(x = x, y = y), "NA or NaN")
  })

  it("rejects NaN in features via matrix interface", {
    x <- matrix(c(1, NaN, 3, 4, 1, 2, 3, 4), ncol = 2)
    y <- c("a", "a", "b", "b")
    expect_error(pptr(x = x, y = y), "NA or NaN")
  })

  it("trains with a constant feature column", {
    df <- data.frame(x1 = rep(5, 6), x2 = c(1, 2, 3, 7, 8, 9), y = c("a", "a", "a", "b", "b", "b"))
    expect_no_error(pptr(y ~ ., data = df, seed = 0))
  })

  it("trains with single observation per group", {
    df <- data.frame(x1 = c(1, 0), x2 = c(0, 1), y = c("a", "b"))
    model <- pptr(y ~ ., data = df, seed = 0)
    expect_s3_class(model, "pptr")
    preds <- predict(model, df)
    expect_length(preds, 2)
  })

  it("trains with minimal dataset (n=2)", {
    df <- data.frame(x1 = c(1, 9), y = c("a", "b"))
    model <- pptr(y ~ ., data = df, seed = 0)
    expect_s3_class(model, "pptr")
    preds <- predict(model, df)
    expect_length(preds, 2)
  })

  it("trains with extreme class imbalance", {
    df <- data.frame(
      x1 = c(0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 90, 91),
      x2 = c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 90, 91),
      y = c(rep("a", 18), rep("b", 2))
    )
    model <- pptr(y ~ ., data = df, seed = 0)
    expect_s3_class(model, "pptr")
    preds <- predict(model, df)
    expect_length(preds, 20)
  })
})
