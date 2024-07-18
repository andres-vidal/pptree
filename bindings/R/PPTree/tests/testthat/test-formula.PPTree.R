Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(PPTree)

describe("formula.PPTree", {
  it("on an object created with the formula interface, returns the formula used to create the model", {
    model <- PPTree(Species ~ ., data = iris)
    expect_equal(formula(model), Species ~ .)
  })

  it("on an object created with the matrix interface, returns NULL", {
    model <- PPTree(x = iris[, 1:4], y = iris[, 5])
    expect_equal(formula(model), NULL)
  })
})
