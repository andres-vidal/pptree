Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

skip_if_not_installed("ggplot2")

describe("plot.pptr importance", {
  it("returns a ggplot object for type = 'importance'", {
    model <- pptr(Type ~ ., data = iris, seed = 42L)
    p <- plot(model, type = "importance")
    expect_s3_class(p, "ggplot")
  })

  it("importance plot orders variables by value", {
    model <- pptr(Type ~ ., data = iris, seed = 42L)
    p <- plot(model, type = "importance")
    pdata <- ggplot2::ggplot_build(p)$layout$panel_params[[1]]$y$get_labels()
    vi <- model$vi$projections
    vnames <- colnames(model$x)
    expected_order <- vnames[order(vi)]
    expect_equal(pdata, expected_order)
  })
})

describe("plot.pprf importance", {
  it("default plot renders importance grid without error", {
    model <- pprf(Type ~ ., data = iris, size = 5, seed = 42L, n_threads = 1)
    expect_no_error(plot(model))
  })

  it("renders importance grid for type = 'importance' without metric", {
    model <- pprf(Type ~ ., data = iris, size = 5, seed = 42L, n_threads = 1)
    expect_no_error(plot(model, type = "importance"))
  })

  it("returns a ggplot for a single importance metric", {
    model <- pprf(Type ~ ., data = iris, size = 5, seed = 42L, n_threads = 1)
    p <- plot(model, metric = "projections")
    expect_s3_class(p, "ggplot")
    p <- plot(model, metric = "weighted")
    expect_s3_class(p, "ggplot")
    p <- plot(model, metric = "permuted")
    expect_s3_class(p, "ggplot")
  })

  it("importance plot orders variables by selected metric", {
    model <- pprf(Type ~ ., data = iris, size = 5, seed = 42L, n_threads = 1)
    p <- plot(model, metric = "permuted")
    pdata <- ggplot2::ggplot_build(p)$layout$panel_params[[1]]$y$get_labels()
    vi <- model$vi$permuted
    vnames <- colnames(model$x)
    expected_order <- vnames[order(vi)]
    expect_equal(pdata, expected_order)
  })
})
