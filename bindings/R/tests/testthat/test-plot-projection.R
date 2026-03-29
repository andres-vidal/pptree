Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

skip_if_not_installed("ggplot2")

describe("plot.pptr projection", {
  it("returns a ggplot object for type = 'projection'", {
    model <- pptr(Type ~ ., data = iris, seed = 0)
    p <- plot(model, type = "projection")
    expect_s3_class(p, "ggplot")
  })

  it("errors for out-of-range node index", {
    model <- pptr(Type ~ ., data = iris, seed = 0)
    expect_error(
      plot(model, type = "projection", node = 100L),
      "out of range"
    )
  })
})

describe("plot.pprf projection", {
  it("returns a ggplot object for type = 'projection'", {
    model <- pprf(Type ~ ., data = iris, size = 5, seed = 0, threads = 1)
    p <- plot(model, type = "projection", tree_index = 1)
    expect_s3_class(p, "ggplot")
  })
})
