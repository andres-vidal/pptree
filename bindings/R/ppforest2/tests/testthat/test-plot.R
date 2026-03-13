Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

skip_if_not_installed("ggplot2")

describe("plot.pptr", {
  it("returns a ggplot object for type = 'structure'", {
    model <- pptr(Type ~ ., data = iris, seed = 42L)
    p <- plot(model, type = "structure")
    expect_s3_class(p, "ggplot")
  })

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

  it("returns a ggplot object for type = 'projection'", {
    model <- pptr(Type ~ ., data = iris, seed = 42L)
    p <- plot(model, type = "projection")
    expect_s3_class(p, "ggplot")
  })

  it("default plot renders mosaic without error", {
    model <- pptr(Type ~ ., data = iris, seed = 42L)
    expect_no_error(plot(model))
  })

  it("errors for out-of-range node index", {
    model <- pptr(Type ~ ., data = iris, seed = 42L)
    expect_error(
      plot(model, type = "projection", node = 100L),
      "out of range"
    )
  })

  it("returns a ggplot object for 2-feature boundary plot", {
    model <- pptr(
      x = iris[, c("Sepal.Length", "Sepal.Width")],
      y = iris$Type,
      seed = 42L
    )
    p <- plot(model, type = "boundaries")
    expect_s3_class(p, "ggplot")
  })

  it("returns a gtable for 3-feature boundary plot (pairwise with stripped panels)", {
    model <- pptr(
      x = iris[, c("Sepal.Length", "Sepal.Width", "Petal.Length")],
      y = iris$Type,
      seed = 42L
    )
    p <- plot(model, type = "boundaries")
    expect_s3_class(p, "gtable")
  })

  it("returns a gtable for pairwise boundary plot with 4 features", {
    model <- pptr(Type ~ ., data = iris, seed = 42L)
    p <- plot(model, type = "boundaries")
    expect_s3_class(p, "gtable")
  })

  it("tree plot uses C++ node data with histograms", {
    model <- pptr(Type ~ ., data = iris, seed = 42L)
    nodes <- ppforest2:::ppforest2_tree_node_data(model, model$x, model$y)
    expect_true(length(nodes) > 0)
    expect_false(nodes[[1]]$is_leaf)
    expect_true(length(nodes[[1]]$projected) > 0)
    expect_true(length(nodes[[1]]$classes) > 0)
  })

  it("boundary plot uses C++ segment computation", {
    model <- pptr(
      x = iris[, c("Sepal.Length", "Sepal.Width")],
      y = iris$Type,
      seed = 42L
    )
    segs <- ppforest2:::ppforest2_boundary_segments(
      model, c(0L, 1L), numeric(0),
      4.0, 8.0, 2.0, 4.5
    )
    expect_s3_class(segs, "data.frame")
    expect_true(nrow(segs) > 0)
    expect_true(all(c("x_start", "y_start", "x_end", "y_end", "depth") %in% names(segs)))
  })

  it("decision regions returns polygon data", {
    model <- pptr(
      x = iris[, c("Sepal.Length", "Sepal.Width")],
      y = iris$Type,
      seed = 42L
    )
    regions <- ppforest2:::ppforest2_decision_regions(
      model, c(0L, 1L), numeric(0),
      4.0, 8.0, 2.0, 4.5
    )
    expect_true(is.list(regions))
    expect_true(length(regions) > 0)
    r <- regions[[1]]
    expect_true(all(c("x", "y", "class") %in% names(r)))
    expect_true(length(r$x) >= 3)
    expect_equal(length(r$x), length(r$y))
  })
})

describe("plot.pprf", {
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

  it("returns a ggplot object for type = 'structure'", {
    model <- pprf(Type ~ ., data = iris, size = 5, seed = 42L, n_threads = 1)
    p <- plot(model, type = "structure", tree_index = 1)
    expect_s3_class(p, "ggplot")
  })

  it("returns a ggplot object for type = 'projection'", {
    model <- pprf(Type ~ ., data = iris, size = 5, seed = 42L, n_threads = 1)
    p <- plot(model, type = "projection", tree_index = 1)
    expect_s3_class(p, "ggplot")
  })

  it("returns a ggplot object for 2-feature boundary plot", {
    model <- pprf(
      x = iris[, c("Sepal.Length", "Sepal.Width")],
      y = iris$Type,
      size = 5, seed = 42L, n_threads = 1
    )
    p <- plot(model, type = "boundaries", tree_index = 1)
    expect_s3_class(p, "ggplot")
  })
})
