Sys.setenv(DEBUG_MODE = "0")

library(testthat)
library(PPTree)

describe("PPForest formula interface", {
  it("returns a object with s3 class PPForest", {
    model <- PPForest(Species ~ ., data = iris)
    expect_s3_class(model, "PPForest")
  })

  it("preserves the observations (x) as a matrix in the returned model", {
    model <- PPForest(Species ~ ., data = iris)
    expected <- model.matrix(Species ~ ., iris, response = TRUE)
    expect_equal(model$x, expected)
  })

  it("preserves the labels (y) as a matrix in the returned model", {
    model <- PPForest(Species ~ ., data = iris)
    expected <- as.matrix(as.numeric(as.factor(iris$Species)))
    expect_equal(model$y, expected)
  })

  it("preserves the classes in the returned model", {
    model <- PPForest(Species ~ ., data = iris)
    expected <- levels(iris$Species)
    expect_equal(model$classes, expected)
  })

  it("preserves the formula in the returned model", {
    model <- PPForest(Species ~ ., data = iris)
    expected <- Species ~ .
    expect_equal(model$formula, expected)
  })
})

describe("PPForest matrix interface", {
  it("returns a object with s3 class PPForest", {
    model <- PPForest(x = iris[, 1:4], y = iris[, 5])
    expect_s3_class(model, "PPForest")
  })

  it("preserves the observations (x) as a matrix in the returned model", {
    model <- PPForest(x = iris[, 1:4], y = iris[, 5])
    expected <- as.matrix(iris[, 1:4])
    expect_equal(model$x, expected)
  })

  it("preserves the labels (y) as a matrix in the returned model", {
    model <- PPForest(x = iris[, 1:4], y = iris[, 5])
    expected <- as.matrix(as.numeric(as.factor(iris$Species)))
    expect_equal(model$y, expected)
  })

  it("preserves the classes in the returned model", {
    model <- PPForest(x = iris[, 1:4], y = iris[, 5])
    expected <- levels(iris$Species)
    expect_equal(model$classes, expected)
  })

  it("does not have a formula", {
    model <- PPForest(x = iris[, 1:4], y = iris[, 5])
    expect_equal(model$formula, NULL)
  })
})

describe("PPForest training spec", {
  it("preserves the lambda parameter in the returned model", {
    model <- PPForest(Species ~ ., data = iris, lambda = 0.5)
    expect_equal(model$training_spec$lambda, 0.5)
  })

  it("the lambda parameter is 0 by default", {
    model <- PPForest(Species ~ ., data = iris)
    expect_equal(model$training_spec$lambda, 0)
  })

  it("preserves the max_retries parameter in the returned model", {
    model <- PPForest(Species ~ ., data = iris, max_retries = 10)
    expect_equal(model$training_spec$max_retries, 10)
  })

  it("the max_retries parameter is 0 by default", {
    model <- PPForest(Species ~ ., data = iris)
    expect_equal(model$training_spec$max_retries, 0)
  })

  it("preserves the n_vars parameter in the returned model", {
    model <- PPForest(Species ~ ., data = iris, n_vars = 2)
    expect_equal(model$training_spec$n_vars, 2)
  })

  it("the n_vars parameter is the number of variables by default", {
    model <- PPForest(x = iris[, 1:4], y = iris[, 5])
    expect_equal(model$training_spec$n_vars, ncol(iris) - 1)
  })

  it("generates as many trees as indicated by the size parameter", {
    model <- PPForest(Species ~ ., data = iris, size = 3)
    expect_equal(length(model$trees), 3)
  })

  it("generates more than one tree if the size parameter is not given", {
    model <- PPForest(Species ~ ., data = iris)
    expect_true(length(model$trees) > 1)
  })

  it("the training strategy is uniform_glda", {
    model <- PPForest(Species ~ ., data = iris)
    expect_equal(model$training_spec$strategy, "uniform_glda")
  })
})

describe("PPForest training data", {
  it("preserves the observations (x) as a matrix without metadata", {
    model <- PPForest(Species ~ ., data = iris)

    expected <- matrix(
      as.vector(model$x),
      nrow = nrow(model$x),
      ncol = ncol(model$x)
    )
    expect_equal(model$training_data$x, expected)
  })

  it("preserves the labels (y) as a vector without metadata", {
    model <- PPForest(Species ~ ., data = iris)
    expected <- as.vector(model$y)
    expect_equal(model$training_data$y, expected)
  })

  it("preserves the classes as a vector without metadata", {
    model <- PPForest(Species ~ ., data = iris)
    expected <- as.integer(unique(iris$Species))
    expect_equal(model$training_data$classes, expected)
  })
})
