% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/PPTree.R
\name{formula.PPTree}
\alias{formula.PPTree}
\title{Extracts the formula used to train a PPTree model.}
\usage{
\method{formula}{PPTree}(x, ...)
}
\arguments{
\item{x}{A PPTree model.}

\item{...}{(unused) other parameters tipically passed to formula}
}
\value{
The formula used to train the model.
}
\description{
Extracts the formula used to train a PPTree model.
}
\examples{
model <- PPTree(Type ~ ., data = iris)
formula(model)
}
