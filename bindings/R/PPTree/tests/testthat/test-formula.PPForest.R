Sys.setenv(DEBUG_MODE = "0")
Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_NUM_THREADS = "1")
Sys.setenv(OMP_THREAD_LIMIT = "1")

library(testthat)
library(PPTree)

describe("formula.PPForest", {
  it("on an object created with the formula interface, returns the formula used to create the model", {
    model <- PPForest(Species ~ ., data = iris)
    expect_equal(formula(model), Species ~ .)
  })

  it("on an object created with the matrix interface, returns NULL", {
    model <- PPForest(x = iris[, 1:4], y = iris[, 5])
    expect_equal(formula(model), NULL)
  })
})
