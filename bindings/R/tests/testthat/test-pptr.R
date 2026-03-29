Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

describe("pptr formula interface", {
  it("returns a object with s3 class pptr", {
    model <- pptr(Type ~ ., data = iris)
    expect_s3_class(model, "pptr")
  })

  it("preserves the observations (x) as a matrix in the returned model", {
    model <- pptr(Type ~ ., data = iris)
    expected <- model.matrix(
      Type ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 1,
      iris,
      response = TRUE
    )
    expect_equal(model$x, expected)
  })

  it("preserves the labels (y) as a matrix in the returned model", {
    model <- pptr(Type ~ ., data = iris)
    expected <- as.matrix(as.numeric(as.factor(iris$Type)))
    expect_equal(model$y, expected)
  })

  it("preserves the groups in the returned model", {
    model <- pptr(Type ~ ., data = iris)
    expected <- levels(iris$Type)
    expect_equal(model$groups, expected)
  })

  it("preserves the formula in the returned model", {
    model <- pptr(Type ~ ., data = iris)
    expect_equal(
      model$formula,
      Type ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 1
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
    expected <- as.matrix(as.numeric(as.factor(iris$Type)))
    expect_equal(model$y, expected)
  })

  it("preserves the groups in the returned model", {
    model <- pptr(x = iris[, 1:4], y = iris[, 5])
    expected <- levels(iris$Type)
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
    model1 <- pptr(Type ~ ., data = iris)
    set.seed(0)
    model2 <- pptr(Type ~ ., data = iris)
    expect_equal(predict(model1, iris), predict(model2, iris))
  })

  it("produces identical predictions with explicit seed", {
    model1 <- pptr(Type ~ ., data = iris, seed = 123L)
    model2 <- pptr(Type ~ ., data = iris, seed = 123L)
    expect_equal(predict(model1, iris), predict(model2, iris))
  })

  it("stores the explicit seed in the model", {
    model <- pptr(Type ~ ., data = iris, seed = 0)
    expect_equal(model$seed, 0)
  })

  it("stores the generated seed when seed is NULL", {
    set.seed(99)
    model <- pptr(Type ~ ., data = iris)
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
    model <- pptr(Type ~ ., data = iris, lambda = 0.5)
    expect_equal(model$training_spec$pp$lambda, 0.5)
  })

  it("the lambda parameter is 0 by default", {
    model <- pptr(Type ~ ., data = iris)
    expect_equal(model$training_spec$pp$lambda, 0)
  })

  it("the pp strategy is pda", {
    model <- pptr(Type ~ ., data = iris)
    expect_equal(model$training_spec$pp$name, "pda")
  })

  it("the dr strategy is noop", {
    model <- pptr(Type ~ ., data = iris)
    expect_equal(model$training_spec$dr$name, "noop")
  })
})

describe("pptr with strategy objects", {
  it("trains with pp = pp_pda()", {
    model <- pptr(Type ~ ., data = iris, pp = pp_pda(0.5), seed = 0)
    expect_s3_class(model, "pptr")
    expect_equal(model$training_spec$pp$lambda, 0.5)
  })

  it("trains with pp = pp_pda(0) (LDA)", {
    model <- pptr(Type ~ ., data = iris, pp = pp_pda(0), seed = 0)
    expect_s3_class(model, "pptr")
    expect_equal(model$training_spec$pp$lambda, 0)
  })

  it("produces identical export as lambda shortcut", {
    model_shortcut <- pptr(Type ~ ., data = iris, lambda = 0.5, seed = 0)
    model_strategy <- pptr(Type ~ ., data = iris, pp = pp_pda(0.5), seed = 0)

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

  it("default tree and fully explicit defaults produce identical export", {
    model_default  <- pptr(Type ~ ., data = iris, seed = 0)
    model_explicit <- pptr(Type ~ ., data = iris, pp = pp_pda(0), sr = sr_mean_of_means(), seed = 0)

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
      pptr(Type ~ ., data = iris, pp = pp_pda(0.5), lambda = 0.3),
      "Cannot use `pp` together with `lambda`")
  })

  it("rejects non-pp_strategy objects", {
    expect_error(
      pptr(Type ~ ., data = iris, pp = list(name = "pda", lambda = 0.5)),
      "pp_strategy")
  })

  it("rejects non-sr_strategy objects", {
    expect_error(
      pptr(Type ~ ., data = iris, sr = list(name = "mean_of_means")),
      "sr_strategy")
  })
})
