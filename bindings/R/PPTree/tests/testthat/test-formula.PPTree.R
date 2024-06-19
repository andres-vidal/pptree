Sys.setenv(DEBUG_MODE = "0")

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
