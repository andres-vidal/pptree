Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(PPTree)

describe("predict.PPTree", {
  describe("on an object created with the formula interface", {
    it("returns a factor with the same length as the input matrix", {
      model <- PPTree(Type ~ ., data = iris)
      predictions <- predict(model, iris)
      expect_equal(length(predictions), nrow(iris))
    })

    it("returns a factor with the same levels as the classes in the model", {
      model <- PPTree(Type ~ ., data = iris)
      predictions <- predict(model, iris)
      expect_equal(levels(predictions), levels(iris$Type))
    })
  })

  describe("on an object created with the matrix interface", {
    it("returns a factor with the same length as the input matrix", {
      x <- crabs[, 2:5]
      x$sex <- as.numeric(as.factor(crabs$sex))
      model <- PPTree(x = x, y = crabs$Type)
      predictions <- predict(model, x)
      expect_equal(length(predictions), nrow(x))
    })

    it("returns a factor with the same levels as the classes in the model", {
      x <- crabs[, 2:5]
      x$sex <- as.numeric(as.factor(crabs$sex))
      model <- PPTree(x = x, y = crabs$Type)
      predictions <- predict(model, x)
      expect_equal(levels(predictions), levels(crabs$Type))
    })
  })
})
