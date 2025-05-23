% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/PPForest.R
\name{predict.PPForest}
\alias{predict.PPForest}
\title{Predicts the labels of a set of observations using a PPForest model.}
\usage{
\method{predict}{PPForest}(object, ...)
}
\arguments{
\item{object}{A PPForest model.}

\item{...}{other parameters tipically passed to predict.}
}
\value{
A matrix containing the predicted labels for each observation.
}
\description{
Predicts the labels of a set of observations using a PPForest model.
}
\examples{
# Example 1: with the `iris` dataset
model <- PPForest(Type ~ ., data = iris)
predict(model, iris)

# Example 2: with the `crabs` dataset
model <- PPForest(Type ~ . - sex + as.numeric(as.factor(sex)), data = crabs)
predict(model, crabs)

}
