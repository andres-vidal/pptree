Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(PPTree)

describe("PPTree formula interface", {
  it("returns a object with s3 class PPTree", {
    model <- PPTree(Type ~ ., data = iris)
    expect_s3_class(model, "PPTree")
  })

  it("preserves the observations (x) as a matrix in the returned model", {
    model <- PPTree(Type ~ ., data = iris)
    expected <- model.matrix(
      Type ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 1,
      iris,
      response = TRUE
    )
    expect_equal(model$x, expected)
  })

  it("preserves the labels (y) as a matrix in the returned model", {
    model <- PPTree(Type ~ ., data = iris)
    expected <- as.matrix(as.numeric(as.factor(iris$Type)))
    expect_equal(model$y, expected)
  })

  it("preserves the classes in the returned model", {
    model <- PPTree(Type ~ ., data = iris)
    expected <- levels(iris$Type)
    expect_equal(model$classes, expected)
  })

  it("preserves the formula in the returned model", {
    model <- PPTree(Type ~ ., data = iris)
    expect_equal(
      model$formula,
      Type ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 1
    )
  })
})

describe("PPTree matrix interface", {
  it("returns a object with s3 class PPTree", {
    model <- PPTree(x = iris[, 1:4], y = iris[, 5])
    expect_s3_class(model, "PPTree")
  })

  it("preserves the observations (x) as a matrix in the returned model", {
    model <- PPTree(x = iris[, 1:4], y = iris[, 5])
    expected <- as.matrix(iris[, 1:4])
    expect_equal(model$x, expected)
  })

  it("preserves the labels (y) as a matrix in the returned model", {
    model <- PPTree(x = iris[, 1:4], y = iris[, 5])
    expected <- as.matrix(as.numeric(as.factor(iris$Type)))
    expect_equal(model$y, expected)
  })

  it("preserves the classes in the returned model", {
    model <- PPTree(x = iris[, 1:4], y = iris[, 5])
    expected <- levels(iris$Type)
    expect_equal(model$classes, expected)
  })

  it("does not have a formula", {
    model <- PPTree(x = iris[, 1:4], y = iris[, 5])
    expect_equal(model$formula, NULL)
  })
})

describe("PPTree training spec", {
  it("preserves the lambda parameter in the returned model", {
    model <- PPTree(Type ~ ., data = iris, lambda = 0.5)
    expect_equal(model$training_spec$lambda, 0.5)
  })

  it("the lambda parameter is 0 by default", {
    model <- PPTree(Type ~ ., data = iris)
    expect_equal(model$training_spec$lambda, 0)
  })

  it("the training strategy is glda", {
    model <- PPTree(Type ~ ., data = iris)
    expect_equal(model$training_spec$strategy, "glda")
  })
})

describe("PPTree training data", {
  it("preserves the observations (x) as a matrix without metadata", {
    model <- PPTree(Type ~ ., data = iris)

    expected <- matrix(
      as.vector(model$x),
      nrow = nrow(model$x),
      ncol = ncol(model$x)
    )
    expect_equal(model$training_data$x, expected, , tolerance = 0.1)
  })

  it("preserves the labels (y) as a vector without metadata", {
    model <- PPTree(Type ~ ., data = iris)
    expected <- as.vector(model$y)
    expect_equal(model$training_data$y, expected)
  })

  it("preserves the classes as a vector without metadata", {
    model <- PPTree(Type ~ ., data = iris)
    expected <- as.integer(unique(iris$Type))
    expect_equal(model$training_data$classes, expected)
  })
})
