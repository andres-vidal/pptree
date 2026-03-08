Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(PPTree)

skip_if_not_installed("jsonlite")

golden_path <- function(...) {
  system.file("golden", ..., package = "PPTree")
}

load_golden <- function(...) {
  path <- golden_path(...)
  skip_if(path == "", message = paste("Golden file not found:", ...))
  jsonlite::fromJSON(path, simplifyVector = TRUE)
}

describe("Golden file: iris tree-glda-s42", {
  golden <- load_golden("iris", "tree-glda-s42.json")

  it("predictions match golden file", {
    model <- PPTree(Type ~ ., data = iris, seed = 42L)
    predicted <- predict(model, iris)
    expected <- factor(model$classes[golden$predictions + 1L], levels = model$classes)
    expect_equal(predicted, expected)
  })

  it("error rate matches golden file", {
    model <- PPTree(Type ~ ., data = iris, seed = 42L)
    predicted <- predict(model, iris)
    actual <- as.numeric(as.factor(iris$Type))
    error_rate <- mean(predicted != iris$Type)
    expect_equal(error_rate, golden$error_rate, tolerance = 1e-3)
  })
})

describe("Golden file: iris forest-glda-t5-s42", {
  golden <- load_golden("iris", "forest-glda-t5-s42.json")

  it("predictions match golden file", {
    model <- PPForest(Type ~ ., data = iris, size = 5, n_vars = 2, seed = 42L, n_threads = 1)
    predicted <- predict(model, iris)
    expected <- factor(model$classes[golden$predictions + 1L], levels = model$classes)
    expect_equal(predicted, expected)
  })

  it("error rate matches golden file", {
    model <- PPForest(Type ~ ., data = iris, size = 5, n_vars = 2, seed = 42L, n_threads = 1)
    predicted <- predict(model, iris)
    error_rate <- mean(predicted != iris$Type)
    expect_equal(error_rate, golden$error_rate, tolerance = 1e-3)
  })

  it("VI projections match golden file", {
    model <- PPForest(Type ~ ., data = iris, size = 5, n_vars = 2, seed = 42L, n_threads = 1)
    expected <- golden$variable_importance$projections
    expect_equal(model$vi$projections, expected, tolerance = 1e-3)
  })

  it("VI weighted projections match golden file", {
    model <- PPForest(Type ~ ., data = iris, size = 5, n_vars = 2, seed = 42L, n_threads = 1)
    expected <- golden$variable_importance$weighted_projections
    expect_equal(model$vi$weighted, expected, tolerance = 1e-3)
  })

  it("VI permuted match golden file", {
    model <- PPForest(Type ~ ., data = iris, size = 5, n_vars = 2, seed = 42L, n_threads = 1)
    expected <- golden$variable_importance$permuted
    expect_equal(model$vi$permuted, expected, tolerance = 1e-3)
  })

  it("vote proportions match golden file", {
    model <- PPForest(Type ~ ., data = iris, size = 5, n_vars = 2, seed = 42L, n_threads = 1)
    probs <- predict(model, iris, type = "prob")
    expected <- golden$vote_proportions
    actual <- unname(as.matrix(probs))
    expect_equal(actual, expected, tolerance = 1e-3)
  })
})
