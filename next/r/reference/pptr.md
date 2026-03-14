# Trains a Project-Pursuit oblique decision tree.

This function trains a Project-Pursuit oblique decision tree using
either a formula and data frame interface or a matrix-based interface.
When using the formula interface, specify the model formula and the data
frame containing the variables. For the matrix-based interface, provide
matrices for the features and labels directly. If `lambda = 0`, the
model is trained using Linear Discriminant Analysis (LDA). If
`lambda > 0`, the model is trained using Penalized Discriminant Analysis
(PDA).

## Usage

``` r
pptr(formula = NULL, data = NULL, x = NULL, y = NULL, lambda = 0, seed = NULL)
```

## Arguments

- formula:

  A formula of the form `y ~ x1 + x2 + ...`, where `y` is a vector of
  labels and `x1`, `x2`, ... are the features.

- data:

  A data frame containing the variables in the formula.

- x:

  A matrix containing the features for each observation.

- y:

  A matrix containing the labels for each observation.

- lambda:

  A regularization parameter. If `lambda = 0`, the model is trained
  using Linear Discriminant Analysis (LDA). If `lambda > 0`, the model
  is trained using Penalized Discriminant Analysis (PDA).

- seed:

  An optional integer seed for reproducibility. If `NULL` (default), a
  seed is drawn from R's RNG, so
  [`set.seed()`](https://rdrr.io/r/base/Random.html) controls
  reproducibility. If an integer is provided, that value is used
  directly.

## Value

A pptr model trained on `x` and `y`.

## See also

[`predict.pptr`](https://andres-vidal.github.io/ppforest2/next/r/reference/predict.pptr.md),
[`formula.pptr`](https://andres-vidal.github.io/ppforest2/next/r/reference/formula.pptr.md),
[`summary.pptr`](https://andres-vidal.github.io/ppforest2/next/r/reference/summary.pptr.md),
[`print.pptr`](https://andres-vidal.github.io/ppforest2/next/r/reference/print.pptr.md),
[`pp_tree`](https://andres-vidal.github.io/ppforest2/next/r/reference/pp_tree.md)
for parsnip integration,
[`vignette("introduction")`](https://andres-vidal.github.io/ppforest2/next/r/articles/introduction.md)
for a tutorial

## Examples

``` r
# Example 1: formula interface with the `iris` dataset
pptr(Type ~ ., data = iris)
#> 
#> Project-Pursuit Oblique Decision Tree:
#> If ([ 0.01 0.04 -0.04 -0.01 ] * x) < 0.06660754:
#>  If ([ 0.04 0.07 -0.09 -0.15 ] * x) < -0.2075133:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 

# Example 2: formula interface with the `iris` dataset with regularization
pptr(Type ~ ., data = iris, lambda = 0.5)
#> 
#> Project-Pursuit Oblique Decision Tree:
#> If ([ 0 -0.04 0.03 0.03 ] * x) < 0.01580044:
#>   Predict: setosa 
#> Else:
#>  If ([ 0 0.03 -0.06 -0.15 ] * x) < -0.4503323:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> 

# Example 3: matrix interface with the `iris` dataset
pptr(x = iris[, 1:4], y = iris[, 5])
#> If ([ 0.01 0.04 -0.04 -0.01 ] * x) < 0.06660754:
#>  If ([ 0.04 0.07 -0.09 -0.15 ] * x) < -0.2075133:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 

# Example 4: matrix interface with the `iris` dataset with regularization
pptr(x = iris[, 1:4], y = iris[, 5], lambda = 0.5)
#> If ([ 0 -0.04 0.03 0.03 ] * x) < 0.01580044:
#>   Predict: setosa 
#> Else:
#>  If ([ 0 0.03 -0.06 -0.15 ] * x) < -0.4503323:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> 

# Example 5: formula interface with the `crabs` dataset
pptr(Type ~ . - sex + as.numeric(as.factor(sex)), data = crabs)
#> 
#> Project-Pursuit Oblique Decision Tree:
#> If ([ 0.04 0.01 0.01 -0.04 0.03 0.02 ] * x) < 0.07781532:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 

# Example 6: formula interface with the `crabs` dataset with regularization
pptr(Type ~ . - sex + as.numeric(as.factor(sex)), data = crabs, lambda = 0.5)
#> 
#> Project-Pursuit Oblique Decision Tree:
#> If ([ 0.01 0 0 0 0.01 0 ] * x) < 0.3234434:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 

# Example 7: matrix interface with the `crabs` dataset
x <- crabs[, 2:5]
x$sex <- as.numeric(as.factor(crabs$sex))
pptr(x = x, y = crabs$Type)
#> If ([ 0.06 0.01 0.03 -0.05 0.02 ] * x) < 0.1153606:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 

# Example 8: matrix interface with the `crabs` dataset with regularization
x <- crabs[, 2:5]
x$sex <- as.numeric(as.factor(crabs$sex))
pptr(x = x, y = crabs$Type, lambda = 0.5)
#> If ([ 0.02 0.01 0 0 0 ] * x) < 0.3518856:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
```
