Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(PPTree)

describe("formula.PPForest", {
  it("on an object created with the formula interface, returns the formula used to create the model", {
    model <- PPForest(Type ~ ., data = iris, n_threads = 1)
    expect_equal(
      formula(model),
      Type ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 1
    )
  })

  it("on an object created with the matrix interface, returns NULL", {
    model <- PPForest(x = iris[, 1:4], y = iris[, 5], n_threads = 1)
    expect_equal(formula(model), NULL)
  })
})
