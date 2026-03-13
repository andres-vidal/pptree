Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

describe("predict.pptr", {
  describe("on an object created with the formula interface", {
    it("returns a factor with the same length as the input matrix", {
      model <- pptr(Type ~ ., data = iris)
      predictions <- predict(model, iris)
      expect_equal(length(predictions), nrow(iris))
    })

    it("returns a factor with the same levels as the classes in the model", {
      model <- pptr(Type ~ ., data = iris)
      predictions <- predict(model, iris)
      expect_equal(levels(predictions), levels(iris$Type))
    })

    it("with new_data parameter returns the same result as positional", {
      model <- pptr(Type ~ ., data = iris)
      pred_positional <- predict(model, iris)
      pred_named <- predict(model, new_data = iris)
      expect_equal(pred_positional, pred_named)
    })
  })

  describe("on an object created with the matrix interface", {
    it("returns a factor with the same length as the input matrix", {
      x <- crabs[, 2:5]
      x$sex <- as.numeric(as.factor(crabs$sex))
      model <- pptr(x = x, y = crabs$Type)
      predictions <- predict(model, x)
      expect_equal(length(predictions), nrow(x))
    })

    it("returns a factor with the same levels as the classes in the model", {
      x <- crabs[, 2:5]
      x$sex <- as.numeric(as.factor(crabs$sex))
      model <- pptr(x = x, y = crabs$Type)
      predictions <- predict(model, x)
      expect_equal(levels(predictions), levels(crabs$Type))
    })
  })

  describe("with type = 'prob'", {
    it("returns a data frame with one column per class", {
      model <- pptr(Type ~ ., data = iris)
      probs <- predict(model, iris, type = "prob")
      expect_true(is.data.frame(probs))
      expect_equal(ncol(probs), length(levels(iris$Type)))
      expect_equal(colnames(probs), levels(iris$Type))
    })

    it("returns exactly one 1.0 per row and the rest 0.0", {
      model <- pptr(Type ~ ., data = iris)
      probs <- predict(model, iris, type = "prob")
      row_sums <- rowSums(probs)
      expect_equal(row_sums, rep(1.0, nrow(iris)))
      expect_true(all(probs == 0 | probs == 1))
    })

    it("the 1.0 column matches the class prediction", {
      model <- pptr(Type ~ ., data = iris)
      class_preds <- predict(model, iris, type = "class")
      prob_preds <- predict(model, iris, type = "prob")
      for (i in seq_len(nrow(iris))) {
        expect_equal(prob_preds[i, as.character(class_preds[i])], 1.0)
      }
    })
  })
})
