Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

describe("pprf variable importance", {
  it("has vi with scale, projections, weighted, permuted", {
    model <- pprf(Species ~ ., data = iris, size = 5, threads = 1)
    # Forest VI: `scale` and `projections` live on `model$vi` (eager, cheap);
    # `weighted` and `permuted` are lazy accessors (not stored as fields).
    expect_true(!is.null(model$vi))
    expect_true("scale" %in% names(model$vi))
    expect_true("projections" %in% names(model$vi))
    p <- ncol(model$x)
    expect_length(model$vi$scale, p)
    expect_length(model$vi$projections, p)
    expect_length(weighted_importance(model), p)
    expect_length(permuted_importance(model), p)
  })

  it("has oob_error", {
    model <- pprf(Species ~ ., data = iris, size = 5, threads = 1)
    err <- oob_error(model)
    expect_true(!is.null(err))
    if (!is.na(err)) {
      expect_gte(err, 0)
      expect_lte(err, 1)
    }
  })

  it("oob_predictions returns all-NA factor when no row has any OOB tree", {
    # Symmetric counterpart to the regression-side test: force every row
    # to be in-bag in every tree, then confirm `oob_predictions` returns
    # an all-NA factor (missing at the factor level, not a raw sentinel
    # level leaking through). The factor's levels must still match
    # `model$groups` — the "all missing" case should not drop levels.
    model <- pprf(Species ~ ., data = iris, size = 3, threads = 1)
    n <- nrow(model$x)
    all_in_bag <- seq_len(n) - 1L
    for (i in seq_along(model$trees)) {
      model$trees[[i]]$sample_indices <- all_in_bag
    }
    if (!is.null(model$.cache) && exists("oob_predictions", envir = model$.cache, inherits = FALSE)) {
      rm("oob_predictions", envir = model$.cache)
    }
    preds <- oob_predictions(model)
    expect_true(is.factor(preds))
    expect_equal(levels(preds), model$groups)
    expect_true(all(is.na(preds)))
  })

  it("oob_error returns NA when no observation has any OOB tree", {
    # Construct a degenerate forest by manually rewriting each tree's
    # sample_indices to cover every training row (all-in-bag). This
    # removes every row from the OOB set, so `oob_error` has no data to
    # compute on and must surface `NA_real_` rather than a sentinel like
    # `-1` that callers might mistake for a (representable but
    # mathematically impossible) error rate.
    model <- pprf(Species ~ ., data = iris, size = 3, threads = 1)
    n <- nrow(model$x)
    all_in_bag <- seq_len(n) - 1L  # 0-based, as stored on the tree
    for (i in seq_along(model$trees)) {
      model$trees[[i]]$sample_indices <- all_in_bag
    }
    # Invalidate the cache (oob_error may have been eagerly primed).
    if (!is.null(model$.cache) && exists("oob_error", envir = model$.cache, inherits = FALSE)) {
      rm("oob_error", envir = model$.cache)
    }
    expect_true(is.na(oob_error(model)))
  })

  it("summary output contains Variable Importance and OOB error", {
    model <- pprf(Species ~ ., data = iris, size = 5, threads = 1)
    out <- capture.output(summary(model))
    expect_true(any(grepl("Variable Importance", out)))
    if (!is.na(oob_error(model))) {
      expect_true(any(grepl("OOB error", out)))
    }
  })
})

describe("pptr variable importance", {
  it("has vi with scale and projections only", {
    model <- pptr(Species ~ ., data = iris)
    expect_true(!is.null(model$vi))
    expect_true("scale" %in% names(model$vi))
    expect_true("projections" %in% names(model$vi))
    expect_false("weighted" %in% names(model$vi))
    expect_false("permuted" %in% names(model$vi))
    p <- ncol(model$x)
    expect_length(model$vi$scale, p)
    expect_length(model$vi$projections, p)
  })

  it("summary output contains Variable Importance", {
    model <- pptr(Species ~ ., data = iris)
    out <- capture.output(summary(model))
    expect_true(any(grepl("Variable Importance", out)))
  })
})
