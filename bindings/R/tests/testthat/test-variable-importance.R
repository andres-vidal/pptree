Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

describe("pprf variable importance", {
  it("has vi with scale, projections, weighted, permuted", {
    model <- pprf(Type ~ ., data = iris, size = 5, threads = 1)
    expect_true(!is.null(model$vi))
    expect_true("scale" %in% names(model$vi))
    expect_true("projections" %in% names(model$vi))
    expect_true("weighted" %in% names(model$vi))
    expect_true("permuted" %in% names(model$vi))
    p <- ncol(model$x)
    expect_length(model$vi$scale, p)
    expect_length(model$vi$projections, p)
    expect_length(model$vi$weighted, p)
    expect_length(model$vi$permuted, p)
  })

  it("has oob_error", {
    model <- pprf(Type ~ ., data = iris, size = 5, threads = 1)
    expect_true(!is.null(model$oob_error))
    if (model$oob_error >= 0) {
      expect_gte(model$oob_error, 0)
      expect_lte(model$oob_error, 1)
    }
  })

  it("summary output contains Variable Importance and OOB error", {
    model <- pprf(Type ~ ., data = iris, size = 5, threads = 1)
    out <- capture.output(summary(model))
    expect_true(any(grepl("Variable Importance", out)))
    if (model$oob_error >= 0) {
      expect_true(any(grepl("OOB error", out)))
    }
  })
})

describe("pptr variable importance", {
  it("has vi with scale and projections only", {
    model <- pptr(Type ~ ., data = iris)
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
    model <- pptr(Type ~ ., data = iris)
    out <- capture.output(summary(model))
    expect_true(any(grepl("Variable Importance", out)))
  })
})
