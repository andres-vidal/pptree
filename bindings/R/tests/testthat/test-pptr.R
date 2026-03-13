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

  it("preserves the classes in the returned model", {
    model <- pptr(Type ~ ., data = iris)
    expected <- levels(iris$Type)
    expect_equal(model$classes, expected)
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

  it("preserves the classes in the returned model", {
    model <- pptr(x = iris[, 1:4], y = iris[, 5])
    expected <- levels(iris$Type)
    expect_equal(model$classes, expected)
  })

  it("does not have a formula", {
    model <- pptr(x = iris[, 1:4], y = iris[, 5])
    expect_equal(model$formula, NULL)
  })
})

describe("pptr reproducibility", {
  it("produces identical predictions with set.seed", {
    set.seed(42)
    model1 <- pptr(Type ~ ., data = iris)
    set.seed(42)
    model2 <- pptr(Type ~ ., data = iris)
    expect_equal(predict(model1, iris), predict(model2, iris))
  })

  it("produces identical predictions with explicit seed", {
    model1 <- pptr(Type ~ ., data = iris, seed = 123L)
    model2 <- pptr(Type ~ ., data = iris, seed = 123L)
    expect_equal(predict(model1, iris), predict(model2, iris))
  })

  it("stores the explicit seed in the model", {
    model <- pptr(Type ~ ., data = iris, seed = 42L)
    expect_equal(model$seed, 42L)
  })

  it("stores the generated seed when seed is NULL", {
    set.seed(99)
    model <- pptr(Type ~ ., data = iris)
    expect_true(is.numeric(model$seed))
    expect_true(model$seed > 0)
  })
})

describe("pptr training spec", {
  it("preserves the lambda parameter in the returned model", {
    model <- pptr(Type ~ ., data = iris, lambda = 0.5)
    expect_equal(model$training_spec$lambda, 0.5)
  })

  it("the lambda parameter is 0 by default", {
    model <- pptr(Type ~ ., data = iris)
    expect_equal(model$training_spec$lambda, 0)
  })

  it("the training strategy is glda", {
    model <- pptr(Type ~ ., data = iris)
    expect_equal(model$training_spec$strategy, "glda")
  })
})
