Sys.setenv(DEBUG_MODE = "0")

library(testthat)
library(PPTree)

describe("predict.PPTree", {
  describe("on an object created with the formula interface", {
    it("returns a factor with the same length as the input matrix", {
      model <- PPTree(Species ~ ., data = iris)
      predictions <- predict(model, iris)
      expect_equal(length(predictions), nrow(iris))
    })

    it("returns a factor with the same levels as the classes in the model", {
      model <- PPTree(Species ~ ., data = iris)
      predictions <- predict(model, iris)
      expect_equal(levels(predictions), levels(iris$Species))
    })
  })

  describe("on an object created with the matrix interface", {
    it("returns a factor with the same length as the input matrix", {
      x <- crabs[, c(2, 4:8)]
      x$sex <- as.numeric(as.factor(x$sex))
      model <- PPTree(x = x, y = crabs[, 1])
      predictions <- predict(model, x)
      expect_equal(length(predictions), nrow(x))
    })

    it("returns a factor with the same levels as the classes in the model", {
      x <- crabs[, c(2, 4:8)]
      x$sex <- as.numeric(as.factor(x$sex))
      model <- PPTree(x = x, y = crabs[, 1])
      predictions <- predict(model, x)
      expect_equal(levels(predictions), levels(crabs$sp))
    })
  })
})
