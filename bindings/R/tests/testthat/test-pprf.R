Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

describe("pprf formula interface", {
  it("returns a object with s3 class pprf", {
    model <- pprf(Type ~ ., data = iris, threads = 1)
    expect_s3_class(model, "pprf")
  })

  it("preserves the observations (x) as a matrix in the returned model", {
    model <- pprf(Type ~ ., data = iris, threads = 1)
    expected <- model.matrix(
      Type ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 1,
      iris,
      response = TRUE
    )
    expect_equal(model$x, expected)
  })

  it("preserves the labels (y) as a matrix in the returned model", {
    model <- pprf(Type ~ ., data = iris, threads = 1)
    expected <- as.matrix(as.numeric(as.factor(iris$Type)))
    expect_equal(model$y, expected)
  })

  it("preserves the groups in the returned model", {
    model <- pprf(Type ~ ., data = iris, threads = 1)
    expected <- levels(iris$Type)
    expect_equal(model$groups, expected)
  })

  it("preserves the formula in the returned model", {
    model <- pprf(Type ~ ., data = iris, threads = 1)

    expect_equal(
      model$formula,
      Type ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 1
    )
  })
})

describe("pprf matrix interface", {
  it("returns a object with s3 class pprf", {
    model <- pprf(x = iris[, 1:4], y = iris[, 5], threads = 1)
    expect_s3_class(model, "pprf")
  })

  it("preserves the observations (x) as a matrix in the returned model", {
    model <- pprf(x = iris[, 1:4], y = iris[, 5], threads = 1)
    expected <- as.matrix(iris[, 1:4])
    expect_equal(model$x, expected)
  })

  it("preserves the labels (y) as a matrix in the returned model", {
    model <- pprf(x = iris[, 1:4], y = iris[, 5], threads = 1)
    expected <- as.matrix(as.numeric(as.factor(iris$Type)))
    expect_equal(model$y, expected)
  })

  it("preserves the groups in the returned model", {
    model <- pprf(x = iris[, 1:4], y = iris[, 5], threads = 1)
    expected <- levels(iris$Type)
    expect_equal(model$groups, expected)
  })

  it("does not have a formula", {
    model <- pprf(x = iris[, 1:4], y = iris[, 5], threads = 1)
    expect_equal(model$formula, NULL)
  })
})

describe("pprf reproducibility", {
  it("produces identical predictions with set.seed", {
    set.seed(0)
    model1 <- pprf(Type ~ ., data = iris, size = 3, threads = 1)
    set.seed(0)
    model2 <- pprf(Type ~ ., data = iris, size = 3, threads = 1)
    expect_equal(predict(model1, iris), predict(model2, iris))
  })

  it("produces identical predictions with explicit seed", {
    model1 <- pprf(Type ~ ., data = iris, size = 3, seed = 123L, threads = 1)
    model2 <- pprf(Type ~ ., data = iris, size = 3, seed = 123L, threads = 1)
    expect_equal(predict(model1, iris), predict(model2, iris))
  })

  it("produces reproducible VI permuted importance", {
    model1 <- pprf(Type ~ ., data = iris, size = 3, seed = 0, threads = 1)
    model2 <- pprf(Type ~ ., data = iris, size = 3, seed = 0, threads = 1)
    expect_equal(model1$vi$permuted, model2$vi$permuted)
  })

  it("stores the explicit seed in the model", {
    model <- pprf(Type ~ ., data = iris, size = 3, seed = 0, threads = 1)
    expect_equal(model$seed, 0)
  })

  it("stores the generated seed when seed is NULL", {
    set.seed(99)
    model <- pprf(Type ~ ., data = iris, size = 3, threads = 1)
    expect_true(is.numeric(model$seed))
    expect_true(model$seed > 0)
  })
})

describe("pprf input validation", {
  it("rejects invalid lambda", {
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], lambda = -1), "between 0 and 1")
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], lambda = 2), "between 0 and 1")
  })

  it("rejects non-integer seed", {
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], seed = 1.5), "integer")
  })

  it("rejects size < 1", {
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], size = 0), "positive integer")
  })

  it("rejects non-integer size", {
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], size = 2.5), "positive integer")
  })

  it("rejects n_vars out of range", {
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], n_vars = 100), "number of features")
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], n_vars = 0), "positive integer")
  })

  it("rejects non-positive threads", {
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], threads = 0), "positive integer")
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], threads = -1), "positive integer")
  })

  it("rejects both n_vars and p_vars", {
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], n_vars = 2, p_vars = 0.5), "not both")
  })

  it("rejects invalid p_vars", {
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], p_vars = 0), "between 0 (exclusive) and 1 (inclusive)", fixed = TRUE)
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], p_vars = 1.5), "between 0 (exclusive) and 1 (inclusive)", fixed = TRUE)
  })

  it("accepts valid p_vars", {
    model <- pprf(x = iris[, 1:4], y = iris[, 5], p_vars = 0.5, size = 2, threads = 1)
    expect_equal(model$training_spec$vars$count, 2)
  })
})

describe("pprf training spec", {
  it("preserves the lambda parameter in the returned model", {
    model <- pprf(Type ~ ., data = iris, lambda = 0.5, threads = 1)
    expect_equal(model$training_spec$pp$lambda, 0.5)
  })

  it("the lambda parameter is 0 by default", {
    model <- pprf(Type ~ ., data = iris, threads = 1)
    expect_equal(model$training_spec$pp$lambda, 0)
  })

  it("preserves the n_vars parameter in the returned model", {
    model <- pprf(Type ~ ., data = iris, n_vars = 2, threads = 1)
    expect_equal(model$training_spec$vars$count, 2)
  })

  it("the n_vars parameter is the number of variables by default", {
    model <- pprf(x = iris[, 1:4], y = iris[, 5], threads = 1)
    expect_equal(model$training_spec$vars$count, ncol(iris) - 1)
  })

  it("generates as many trees as indicated by the size parameter", {
    model <- pprf(Type ~ ., data = iris, size = 3, threads = 1)
    expect_equal(length(model$trees), 3)
  })

  it("generates more than one tree if the size parameter is not given", {
    model <- pprf(Type ~ ., data = iris, threads = 1)
    expect_true(length(model$trees) > 1)
  })

  it("the pp strategy is pda", {
    model <- pprf(Type ~ ., data = iris, threads = 1)
    expect_equal(model$training_spec$pp$name, "pda")
  })

  it("the vars strategy is uniform", {
    model <- pprf(Type ~ ., data = iris, threads = 1)
    expect_equal(model$training_spec$vars$name, "uniform")
  })

  it("the stop strategy is pure_node", {
    model <- pprf(Type ~ ., data = iris, threads = 1)
    expect_equal(model$training_spec$stop$name, "pure_node")
  })

  it("the binarize strategy is largest_gap", {
    model <- pprf(Type ~ ., data = iris, threads = 1)
    expect_equal(model$training_spec$binarize$name, "largest_gap")
  })

  it("the partition strategy is by_group", {
    model <- pprf(Type ~ ., data = iris, threads = 1)
    expect_equal(model$training_spec$partition$name, "by_group")
  })

  it("the leaf strategy is majority_vote", {
    model <- pprf(Type ~ ., data = iris, threads = 1)
    expect_equal(model$training_spec$leaf$name, "majority_vote")
  })
})

describe("pprf with strategy objects", {
  it("trains with pp and vars", {
    model <- pprf(Type ~ ., data = iris, size = 2, pp = pp_pda(0.5), vars = vars_uniform(n_vars = 2), seed = 0, threads = 1)
    expect_s3_class(model, "pprf")
    expect_equal(model$training_spec$pp$lambda, 0.5)
    expect_equal(model$training_spec$vars$count, 2)
  })

  it("trains with vars using p_vars", {
    model <- pprf(Type ~ ., data = iris, size = 2, vars = vars_uniform(p_vars = 0.5), seed = 0, threads = 1)
    expect_s3_class(model, "pprf")
    expect_equal(model$training_spec$vars$count, 2)  # 0.5 * 4 = 2
  })

  it("produces identical export as shortcut params", {
    model_shortcut <- pprf(Type ~ ., data = iris, size = 3, lambda = 0.5, n_vars = 2, seed = 0, threads = 1)
    model_strategy <- pprf(Type ~ ., data = iris, size = 3, pp = pp_pda(0.5), vars = vars_uniform(n_vars = 2), seed = 0, threads = 1)

    path_shortcut <- tempfile(fileext = ".json")
    path_strategy <- tempfile(fileext = ".json")
    save_json(model_shortcut, path_shortcut)
    save_json(model_strategy, path_strategy)

    j_shortcut <- jsonlite::fromJSON(readLines(path_shortcut, warn = FALSE))
    j_strategy <- jsonlite::fromJSON(readLines(path_strategy, warn = FALSE))

    j_shortcut$training_duration_ms <- NULL
    j_strategy$training_duration_ms <- NULL
    expect_equal(j_shortcut, j_strategy)
  })

  it("p_vars shortcut and vars_uniform(p_vars) produce identical export", {
    model_shortcut <- pprf(Type ~ ., data = iris, size = 3, p_vars = 0.5, seed = 0, threads = 1)
    model_strategy <- pprf(Type ~ ., data = iris, size = 3, vars = vars_uniform(p_vars = 0.5), seed = 0, threads = 1)

    path_shortcut <- tempfile(fileext = ".json")
    path_strategy <- tempfile(fileext = ".json")
    save_json(model_shortcut, path_shortcut)
    save_json(model_strategy, path_strategy)

    j_shortcut <- jsonlite::fromJSON(readLines(path_shortcut, warn = FALSE))
    j_strategy <- jsonlite::fromJSON(readLines(path_strategy, warn = FALSE))

    j_shortcut$training_duration_ms <- NULL
    j_strategy$training_duration_ms <- NULL
    expect_equal(j_shortcut, j_strategy)
  })

  it("errors when mixing pp and lambda", {
    expect_error(
      pprf(Type ~ ., data = iris, pp = pp_pda(0.5), lambda = 0.3),
      "Cannot use `pp` together with `lambda`")
  })

  it("errors when mixing vars and n_vars", {
    expect_error(
      pprf(Type ~ ., data = iris, vars = vars_uniform(n_vars = 2), n_vars = 3),
      "Cannot use `vars` together with `n_vars`/`p_vars`")
  })

  it("errors when mixing vars and p_vars", {
    expect_error(
      pprf(Type ~ ., data = iris, vars = vars_uniform(n_vars = 2), p_vars = 0.5),
      "Cannot use `vars` together with `n_vars`/`p_vars`")
  })

  it("rejects non-vars_strategy objects for vars", {
    expect_error(
      pprf(Type ~ ., data = iris, vars = list(name = "uniform", count = 2)),
      "vars_strategy object")
  })

  it("works with parsnip engine args pattern", {
    model <- pprf(Type ~ ., data = iris, size = 2, pp = pp_pda(0.5), vars = vars_uniform(n_vars = 2), seed = 0, threads = 1)
    preds <- predict(model, iris)
    expect_length(preds, 150)
  })

  it("trains with vars_all (no variable subsampling)", {
    model <- pprf(Type ~ ., data = iris, size = 2, vars = vars_all(), seed = 0, threads = 1)
    expect_s3_class(model, "pprf")
    expect_equal(model$training_spec$vars$name, "all")
  })

  it("training spec reflects strategy objects, not just defaults", {
    model <- pprf(Type ~ ., data = iris, size = 2, pp = pp_pda(0.7), vars = vars_uniform(n_vars = 3), cutpoint = cutpoint_mean_of_means(), seed = 0, threads = 1)
    expect_equal(model$training_spec$pp$name, "pda")
    expect_equal(model$training_spec$pp$lambda, 0.7, tolerance = 1e-5)
    expect_equal(model$training_spec$vars$name, "uniform")
    expect_equal(model$training_spec$vars$count, 3)
    expect_equal(model$training_spec$cutpoint$name, "mean_of_means")
  })

  it("trains with explicit stop, binarize, partition, leaf", {
    model <- pprf(Type ~ ., data = iris, size = 2, stop = stop_pure_node(), binarize = binarize_largest_gap(), partition = partition_by_group(), leaf = leaf_majority_vote(), seed = 0, threads = 1)
    expect_s3_class(model, "pprf")
    expect_equal(model$training_spec$stop$name, "pure_node")
    expect_equal(model$training_spec$binarize$name, "largest_gap")
    expect_equal(model$training_spec$partition$name, "by_group")
    expect_equal(model$training_spec$leaf$name, "majority_vote")
  })

  it("rejects non-stop_strategy objects", {
    expect_error(
      pprf(Type ~ ., data = iris, stop = list(name = "pure_node")),
      "stop_strategy")
  })

  it("rejects non-binarize_strategy objects", {
    expect_error(
      pprf(Type ~ ., data = iris, binarize = list(name = "largest_gap")),
      "binarize_strategy")
  })

  it("rejects non-partition_strategy objects", {
    expect_error(
      pprf(Type ~ ., data = iris, partition = list(name = "by_group")),
      "partition_strategy")
  })

  it("rejects non-leaf_strategy objects", {
    expect_error(
      pprf(Type ~ ., data = iris, leaf = list(name = "majority_vote")),
      "leaf_strategy")
  })
})

describe("pprf edge cases", {
  it("rejects NA in features via matrix interface", {
    x <- matrix(c(1, NA, 3, 4, 1, 2, 3, 4), ncol = 2)
    y <- c("a", "a", "b", "b")
    expect_error(pprf(x = x, y = y, size = 3, threads = 1), "NA or NaN")
  })

  it("rejects NaN in features via matrix interface", {
    x <- matrix(c(1, NaN, 3, 4, 1, 2, 3, 4), ncol = 2)
    y <- c("a", "a", "b", "b")
    expect_error(pprf(x = x, y = y, size = 3, threads = 1), "NA or NaN")
  })

  it("trains with a constant feature column", {
    df <- data.frame(x1 = rep(5, 6), x2 = c(1, 2, 3, 7, 8, 9), y = c("a", "a", "a", "b", "b", "b"))
    expect_no_error(pprf(y ~ ., data = df, size = 3, seed = 0, threads = 1))
  })

  it("trains with single observation per group", {
    df <- data.frame(x1 = c(1, 0), x2 = c(0, 1), y = c("a", "b"))
    model <- pprf(y ~ ., data = df, size = 3, seed = 0, threads = 1)
    expect_s3_class(model, "pprf")
    preds <- predict(model, df)
    expect_length(preds, 2)
  })

  it("trains with minimal dataset (n=2)", {
    df <- data.frame(x1 = c(1, 9), y = c("a", "b"))
    model <- pprf(y ~ ., data = df, size = 3, seed = 0, threads = 1)
    expect_s3_class(model, "pprf")
    preds <- predict(model, df)
    expect_length(preds, 2)
  })

  it("trains with extreme class imbalance", {
    df <- data.frame(
      x1 = c(0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 90, 91),
      x2 = c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 90, 91),
      y = c(rep("a", 18), rep("b", 2))
    )
    model <- pprf(y ~ ., data = df, size = 5, seed = 0, threads = 1)
    expect_s3_class(model, "pprf")
    preds <- predict(model, df)
    expect_length(preds, 20)
  })
})
