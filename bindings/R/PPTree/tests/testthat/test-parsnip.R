Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(PPTree)

skip_if_not_installed("parsnip")

library(parsnip)

describe("parsnip integration", {
  describe("pp_forest", {
    it("can be created as a parsnip model specification", {
      spec <- pp_forest() |>
        set_engine("PPTree") |>
        set_mode("classification")
      expect_s3_class(spec, "pp_forest")
    })

    it("can be fit using a parsnip workflow", {
      spec <- pp_forest() |>
        set_engine("PPTree") |>
        set_mode("classification")
      fit <- spec |> fit(Type ~ ., data = iris)
      expect_s3_class(fit, "model_fit")
      expect_s3_class(fit$fit, "PPForest")
    })

    it("can predict classes via parsnip", {
      spec <- pp_forest() |>
        set_engine("PPTree") |>
        set_mode("classification")
      fit <- spec |> fit(Type ~ ., data = iris)
      preds <- predict(fit, iris)
      expect_s3_class(preds, "tbl_df")
      expect_equal(nrow(preds), nrow(iris))
      expect_true(".pred_class" %in% colnames(preds))
    })

    it("can predict probabilities via parsnip", {
      spec <- pp_forest() |>
        set_engine("PPTree") |>
        set_mode("classification")
      fit <- spec |> fit(Type ~ ., data = iris)
      preds <- predict(fit, iris, type = "prob")
      expect_s3_class(preds, "tbl_df")
      expect_equal(nrow(preds), nrow(iris))
      prob_cols <- grep("^\\.pred_", colnames(preds), value = TRUE)
      expect_equal(length(prob_cols), length(levels(iris$Type)))
    })
  })

  describe("pp_tree", {
    it("can be created as a parsnip model specification", {
      spec <- pp_tree() |>
        set_engine("PPTree") |>
        set_mode("classification")
      expect_s3_class(spec, "pp_tree")
    })

    it("can be fit using a parsnip workflow", {
      spec <- pp_tree() |>
        set_engine("PPTree") |>
        set_mode("classification")
      fit <- spec |> fit(Type ~ ., data = iris)
      expect_s3_class(fit, "model_fit")
      expect_s3_class(fit$fit, "PPTree")
    })

    it("can predict classes via parsnip", {
      spec <- pp_tree() |>
        set_engine("PPTree") |>
        set_mode("classification")
      fit <- spec |> fit(Type ~ ., data = iris)
      preds <- predict(fit, iris)
      expect_s3_class(preds, "tbl_df")
      expect_equal(nrow(preds), nrow(iris))
      expect_true(".pred_class" %in% colnames(preds))
    })

    it("can predict probabilities via parsnip", {
      spec <- pp_tree() |>
        set_engine("PPTree") |>
        set_mode("classification")
      fit <- spec |> fit(Type ~ ., data = iris)
      preds <- predict(fit, iris, type = "prob")
      expect_s3_class(preds, "tbl_df")
      expect_equal(nrow(preds), nrow(iris))
    })
  })
})
