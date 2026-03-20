Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite is required for reproducibility tests")
}

golden_path <- function(...) {
  system.file("golden", ..., package = "ppforest2")
}

load_golden <- function(...) {
  path <- golden_path(...)
  if (path == "" || !file.exists(path)) {
    stop(paste("Golden file not found:", ...))
  }
  # simplifyVector = FALSE preserves nested tree structure for forests.
  # Flat arrays are converted to vectors/matrices in comparison helpers.
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

# Convert a JSON list (from simplifyVector=FALSE) to a numeric/integer vector.
as_vec <- function(x) unlist(x)

# Convert a JSON list-of-lists to a matrix (row-major).
as_mat <- function(x) {
  do.call(rbind, lapply(x, unlist))
}

# Golden files use C++ class ordering (order of first appearance in CSV).
# R uses alphabetical factor levels by default. To ensure the R model is
# trained on identically-ordered data, we reorder the factor levels to
# match the golden file before training.
prepare_data <- function(data, golden) {
  golden_classes <- as_vec(golden$meta$classes)
  data$Type <- factor(data$Type, levels = golden_classes)
  data
}

# ---------------------------------------------------------------------------
# Model structure comparison
# ---------------------------------------------------------------------------

# Normalize a node to a canonical representation for comparison.
# R model nodes use 1-based class indices; golden JSON uses 0-based.
# Both are converted to 0-based with consistent numeric types.
normalize_node <- function(node, r_to_cpp = FALSE) {
  offset <- if (r_to_cpp) 1L else 0L
  if (!is.null(node$value)) {
    list(value = as.numeric(node$value) - offset)
  } else {
    list(
      projector      = as.numeric(unlist(node$projector)),
      threshold      = as.numeric(node$threshold),
      pp_index_value = as.numeric(node$pp_index_value),
      classes        = sort(as.integer(unlist(node$classes)) - offset),
      lower          = normalize_node(node$lower, r_to_cpp),
      upper          = normalize_node(node$upper, r_to_cpp)
    )
  }
}

compare_tree <- function(actual_root, expected_root, info = "") {
  actual <- normalize_node(actual_root, r_to_cpp = TRUE)
  expected <- normalize_node(expected_root, r_to_cpp = FALSE)
  expect_equal(actual, expected, tolerance = 1e-3, info = info)
}

# ---------------------------------------------------------------------------
# Output comparison helpers
# ---------------------------------------------------------------------------

compare_predictions <- function(model, golden, data) {
  predicted <- predict(model, data)
  golden_classes <- as_vec(golden$meta$classes)
  preds <- as.integer(as_vec(golden$predictions))
  expected <- factor(golden_classes[preds + 1L], levels = levels(predicted))
  expect_equal(predicted, expected)
}

compare_error_rate <- function(model, golden, data, response) {
  predicted <- predict(model, data)
  error_rate <- mean(predicted != response)
  expect_equal(error_rate, golden$error_rate, tolerance = 1e-3)
}

compare_confusion_matrix <- function(model, golden, data, response) {
  predicted <- predict(model, data)
  expected_cm <- golden$confusion_matrix

  n_classes <- length(model$classes)
  # Align predicted levels with response levels (golden class ordering)
  predicted <- factor(predicted, levels = levels(response))
  actual_idx <- as.integer(response)
  pred_idx <- as.integer(predicted)

  cm <- matrix(0L, nrow = n_classes, ncol = n_classes)
  for (i in seq_along(actual_idx)) {
    cm[actual_idx[i], pred_idx[i]] <- cm[actual_idx[i], pred_idx[i]] + 1L
  }

  expect_equal(cm, as_mat(expected_cm$matrix))
  expect_equal(as.integer(as_vec(expected_cm$labels)), seq(0L, n_classes - 1L))

  class_errors <- numeric(n_classes)
  for (i in seq_len(n_classes)) {
    row_total <- sum(cm[i, ])
    if (row_total > 0) {
      class_errors[i] <- 1 - cm[i, i] / row_total
    }
  }
  expect_equal(class_errors, as.numeric(as_vec(expected_cm$class_errors)),
    tolerance = 1e-3)
}

compare_vote_proportions <- function(model, golden, data) {
  probs <- predict(model, data, type = "prob")
  actual <- unname(as.matrix(probs))
  expect_equal(actual, as_mat(golden$vote_proportions), tolerance = 1e-3)
}

compare_vi <- function(model, golden, key, r_field) {
  expected <- as.numeric(as_vec(golden$variable_importance[[key]]))
  actual <- model$vi[[r_field]]
  expect_equal(actual, expected, tolerance = 1e-3)
}

# ---------------------------------------------------------------------------
# Tree: iris tree-pda-s42
# ---------------------------------------------------------------------------

describe("Reproducibility: iris tree-pda-s42", {
  golden <- load_golden("iris", "tree-pda-s42.json")
  d <- prepare_data(iris, golden)
  model <- pptr(Type ~ ., data = d, seed = 42L)

  it("model structure matches golden file", {
    compare_tree(model$root, golden$model$root)
  })

  it("predictions match golden file", {
    compare_predictions(model, golden, d)
  })

  it("error rate matches golden file", {
    compare_error_rate(model, golden, d, d$Type)
  })

  it("confusion matrix matches golden file", {
    compare_confusion_matrix(model, golden, d, d$Type)
  })

  it("VI projections match golden file", {
    compare_vi(model, golden, "projections", "projections")
  })
})

# ---------------------------------------------------------------------------
# Tree: crab tree-pda-s42
# ---------------------------------------------------------------------------

describe("Reproducibility: crab tree-pda-s42", {
  golden <- load_golden("crab", "tree-pda-s42.json")
  d <- prepare_data(crab, golden)
  model <- pptr(Type ~ ., data = d, seed = 42L)

  it("model structure matches golden file", {
    compare_tree(model$root, golden$model$root)
  })

  it("predictions match golden file", {
    compare_predictions(model, golden, d)
  })

  it("error rate matches golden file", {
    compare_error_rate(model, golden, d, d$Type)
  })

  it("confusion matrix matches golden file", {
    compare_confusion_matrix(model, golden, d, d$Type)
  })

  it("VI projections match golden file", {
    compare_vi(model, golden, "projections", "projections")
  })
})

# ---------------------------------------------------------------------------
# Forest: iris forest-pda-t5-s42
# ---------------------------------------------------------------------------

describe("Reproducibility: iris forest-pda-t5-s42", {
  golden <- load_golden("iris", "forest-pda-t5-s42.json")
  d <- prepare_data(iris, golden)
  model <- pprf(Type ~ ., data = d, size = 5, n_vars = 2, seed = 42L, n_threads = 1)

  it("model structure matches golden file", {
    for (i in seq_along(model$trees)) {
      compare_tree(model$trees[[i]]$root, golden$model$trees[[i]]$root,
        info = paste("tree", i))
    }
  })

  it("predictions match golden file", {
    compare_predictions(model, golden, d)
  })

  it("error rate matches golden file", {
    compare_error_rate(model, golden, d, d$Type)
  })

  it("confusion matrix matches golden file", {
    compare_confusion_matrix(model, golden, d, d$Type)
  })

  it("OOB error matches golden file", {
    expect_equal(model$oob_error, golden$oob_error, tolerance = 1e-3)
  })

  it("VI projections match golden file", {
    compare_vi(model, golden, "projections", "projections")
  })

  it("VI weighted projections match golden file", {
    compare_vi(model, golden, "weighted_projections", "weighted")
  })

  it("VI permuted match golden file", {
    compare_vi(model, golden, "permuted", "permuted")
  })

  it("vote proportions match golden file", {
    compare_vote_proportions(model, golden, d)
  })
})

# ---------------------------------------------------------------------------
# Forest: iris forest-pda-l05-t5-s42
# ---------------------------------------------------------------------------

describe("Reproducibility: iris forest-pda-l05-t5-s42", {
  golden <- load_golden("iris", "forest-pda-l05-t5-s42.json")
  d <- prepare_data(iris, golden)
  model <- pprf(Type ~ ., data = d, size = 5, n_vars = 2, lambda = 0.5, seed = 42L, n_threads = 1)

  it("model structure matches golden file", {
    for (i in seq_along(model$trees)) {
      compare_tree(model$trees[[i]]$root, golden$model$trees[[i]]$root,
        info = paste("tree", i))
    }
  })

  it("predictions match golden file", {
    compare_predictions(model, golden, d)
  })

  it("error rate matches golden file", {
    compare_error_rate(model, golden, d, d$Type)
  })

  it("confusion matrix matches golden file", {
    compare_confusion_matrix(model, golden, d, d$Type)
  })

  it("OOB error matches golden file", {
    expect_equal(model$oob_error, golden$oob_error, tolerance = 1e-3)
  })

  it("VI projections match golden file", {
    compare_vi(model, golden, "projections", "projections")
  })

  it("VI weighted projections match golden file", {
    compare_vi(model, golden, "weighted_projections", "weighted")
  })

  it("VI permuted match golden file", {
    compare_vi(model, golden, "permuted", "permuted")
  })

  it("vote proportions match golden file", {
    compare_vote_proportions(model, golden, d)
  })
})

# ---------------------------------------------------------------------------
# Forest: crab forest-pda-t10-s42
# ---------------------------------------------------------------------------

describe("Reproducibility: crab forest-pda-t10-s42", {
  golden <- load_golden("crab", "forest-pda-t10-s42.json")
  d <- prepare_data(crab, golden)
  model <- pprf(Type ~ ., data = d, size = 10, n_vars = 3, seed = 42L, n_threads = 1)

  it("model structure matches golden file", {
    for (i in seq_along(model$trees)) {
      compare_tree(model$trees[[i]]$root, golden$model$trees[[i]]$root,
        info = paste("tree", i))
    }
  })

  it("predictions match golden file", {
    compare_predictions(model, golden, d)
  })

  it("error rate matches golden file", {
    compare_error_rate(model, golden, d, d$Type)
  })

  it("confusion matrix matches golden file", {
    compare_confusion_matrix(model, golden, d, d$Type)
  })

  it("OOB error matches golden file", {
    expect_equal(model$oob_error, golden$oob_error, tolerance = 1e-3)
  })

  it("VI projections match golden file", {
    compare_vi(model, golden, "projections", "projections")
  })

  it("VI weighted projections match golden file", {
    compare_vi(model, golden, "weighted_projections", "weighted")
  })

  it("VI permuted match golden file", {
    compare_vi(model, golden, "permuted", "permuted")
  })

  it("vote proportions match golden file", {
    compare_vote_proportions(model, golden, d)
  })
})

# ---------------------------------------------------------------------------
# Forest: wine forest-pda-t10-s42
# ---------------------------------------------------------------------------

describe("Reproducibility: wine forest-pda-t10-s42", {
  golden <- load_golden("wine", "forest-pda-t10-s42.json")
  d <- prepare_data(wine, golden)
  model <- pprf(Type ~ ., data = d, size = 10, n_vars = 4, seed = 42L, n_threads = 1)

  it("model structure matches golden file", {
    for (i in seq_along(model$trees)) {
      compare_tree(model$trees[[i]]$root, golden$model$trees[[i]]$root,
        info = paste("tree", i))
    }
  })

  it("predictions match golden file", {
    compare_predictions(model, golden, d)
  })

  it("error rate matches golden file", {
    compare_error_rate(model, golden, d, d$Type)
  })

  it("confusion matrix matches golden file", {
    compare_confusion_matrix(model, golden, d, d$Type)
  })

  it("OOB error matches golden file", {
    expect_equal(model$oob_error, golden$oob_error, tolerance = 1e-3)
  })

  it("VI projections match golden file", {
    compare_vi(model, golden, "projections", "projections")
  })

  it("VI weighted projections match golden file", {
    compare_vi(model, golden, "weighted_projections", "weighted")
  })

  it("VI permuted match golden file", {
    compare_vi(model, golden, "permuted", "permuted")
  })

  it("vote proportions match golden file", {
    compare_vote_proportions(model, golden, d)
  })
})

# ---------------------------------------------------------------------------
# Forest: glass forest-pda-t10-s42
# ---------------------------------------------------------------------------

describe("Reproducibility: glass forest-pda-t10-s42", {
  golden <- load_golden("glass", "forest-pda-t10-s42.json")
  d <- prepare_data(glass, golden)
  model <- pprf(Type ~ ., data = d, size = 10, n_vars = 3, seed = 42L, n_threads = 1)

  it("model structure matches golden file", {
    for (i in seq_along(model$trees)) {
      compare_tree(model$trees[[i]]$root, golden$model$trees[[i]]$root,
        info = paste("tree", i))
    }
  })

  it("predictions match golden file", {
    compare_predictions(model, golden, d)
  })

  it("error rate matches golden file", {
    compare_error_rate(model, golden, d, d$Type)
  })

  it("confusion matrix matches golden file", {
    compare_confusion_matrix(model, golden, d, d$Type)
  })

  it("OOB error matches golden file", {
    expect_equal(model$oob_error, golden$oob_error, tolerance = 1e-3)
  })

  it("VI projections match golden file", {
    compare_vi(model, golden, "projections", "projections")
  })

  it("VI weighted projections match golden file", {
    compare_vi(model, golden, "weighted_projections", "weighted")
  })

  it("VI permuted match golden file", {
    compare_vi(model, golden, "permuted", "permuted")
  })

  it("vote proportions match golden file", {
    compare_vote_proportions(model, golden, d)
  })
})
