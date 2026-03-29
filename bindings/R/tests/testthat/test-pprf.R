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
    expect_equal(model$training_spec$dr$n_vars, 2)
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
    expect_equal(model$training_spec$dr$n_vars, 2)
  })

  it("the n_vars parameter is the number of variables by default", {
    model <- pprf(x = iris[, 1:4], y = iris[, 5], threads = 1)
    expect_equal(model$training_spec$dr$n_vars, ncol(iris) - 1)
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

  it("the dr strategy is uniform", {
    model <- pprf(Type ~ ., data = iris, threads = 1)
    expect_equal(model$training_spec$dr$name, "uniform")
  })
})

describe("pprf with strategy objects", {
  it("trains with pp and dr", {
    model <- pprf(Type ~ ., data = iris, size = 2, pp = pp_pda(0.5), dr = dr_uniform(n_vars = 2), seed = 0, threads = 1)
    expect_s3_class(model, "pprf")
    expect_equal(model$training_spec$pp$lambda, 0.5)
    expect_equal(model$training_spec$dr$n_vars, 2)
  })

  it("trains with dr using p_vars", {
    model <- pprf(Type ~ ., data = iris, size = 2, dr = dr_uniform(p_vars = 0.5), seed = 0, threads = 1)
    expect_s3_class(model, "pprf")
    expect_equal(model$training_spec$dr$n_vars, 2)  # 0.5 * 4 = 2
  })

  it("produces same results as shortcut params", {
    model_shortcut <- pprf(Type ~ ., data = iris, size = 3, lambda = 0.5, n_vars = 2, seed = 0, threads = 1)
    model_strategy <- pprf(Type ~ ., data = iris, size = 3, pp = pp_pda(0.5), dr = dr_uniform(n_vars = 2), seed = 0, threads = 1)
    expect_equal(predict(model_shortcut, iris), predict(model_strategy, iris))
  })

  it("errors when mixing pp and lambda", {
    expect_error(
      pprf(Type ~ ., data = iris, pp = pp_pda(0.5), lambda = 0.3),
      "Cannot use `pp` together with `lambda`")
  })

  it("errors when mixing dr and n_vars", {
    expect_error(
      pprf(Type ~ ., data = iris, dr = dr_uniform(n_vars = 2), n_vars = 3),
      "Cannot use `dr` together with `n_vars`/`p_vars`")
  })

  it("errors when mixing dr and p_vars", {
    expect_error(
      pprf(Type ~ ., data = iris, dr = dr_uniform(n_vars = 2), p_vars = 0.5),
      "Cannot use `dr` together with `n_vars`/`p_vars`")
  })

  it("rejects non-dr_strategy objects", {
    expect_error(
      pprf(Type ~ ., data = iris, dr = list(name = "uniform", n_vars = 2)),
      "dr_strategy")
  })

  it("works with parsnip engine args pattern", {
    model <- pprf(Type ~ ., data = iris, size = 2, pp = pp_pda(0.5), dr = dr_uniform(n_vars = 2), seed = 0, threads = 1)
    preds <- predict(model, iris)
    expect_length(preds, 150)
  })
})
