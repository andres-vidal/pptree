# Trains a Project-Pursuit oblique decision tree.

This function trains a Project-Pursuit oblique decision tree (PPTree)
using either a formula and data frame interface or a matrix-based
interface. When using the formula interface, specify the model formula
and the data frame containing the variables. For the matrix-based
interface, provide matrices for the features and labels directly. If
`lambda = 0`, the model is trained using Linear Discriminant Analysis
(LDA). If `lambda > 0`, the model is trained using Penalized
Discriminant Analysis (PDA).

## Usage

``` r
PPTree(
  formula = NULL,
  data = NULL,
  x = NULL,
  y = NULL,
  lambda = 0,
  seed = NULL
)
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

A PPTree model trained on `x` and `y`.

## See also

[`predict.PPTree`](https://andres-vidal.github.io/pptree/main/r/reference/predict.PPTree.md),
[`formula.PPTree`](https://andres-vidal.github.io/pptree/main/r/reference/formula.PPTree.md),
[`summary.PPTree`](https://andres-vidal.github.io/pptree/main/r/reference/summary.PPTree.md),
[`print.PPTree`](https://andres-vidal.github.io/pptree/main/r/reference/print.PPTree.md),
[`pp_tree`](https://andres-vidal.github.io/pptree/main/r/reference/pp_tree.md)
for parsnip integration

## Examples

``` r
# Example 1: formula interface with the `iris` dataset
PPTree(Type ~ ., data = iris)
#> 
#> Project-Pursuit Oblique Decision Tree:
#> If ([ 0.19 0.71 -0.66 -0.17 ] * x) < 1.070804:
#>  If ([ 0.23 0.36 -0.44 -0.79 ] * x) < -1.062904:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 

# Example 2: formula interface with the `iris` dataset with regularization
PPTree(Type ~ ., data = iris, lambda = 0.5)
#> 
#> Project-Pursuit Oblique Decision Tree:
#> If ([ 0.07 -0.68 0.45 0.58 ] * x) < 0.2732363:
#>   Predict: setosa 
#> Else:
#>  If ([ 0.01 0.16 -0.34 -0.93 ] * x) < -2.714732:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> 

# Example 3: matrix interface with the `iris` dataset
PPTree(x = iris[, 1:4], y = iris[, 5])
#> If ([ 0.19 0.71 -0.66 -0.17 ] * x) < 1.070804:
#>  If ([ 0.23 0.36 -0.44 -0.79 ] * x) < -1.062904:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 

# Example 4: matrix interface with the `iris` dataset with regularization
PPTree(x = iris[, 1:4], y = iris[, 5], lambda = 0.5)
#> If ([ 0.07 -0.68 0.45 0.58 ] * x) < 0.2732363:
#>   Predict: setosa 
#> Else:
#>  If ([ 0.01 0.16 -0.34 -0.93 ] * x) < -2.714732:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> 

# Example 5: formula interface with the `crabs` dataset
PPTree(Type ~ . - sex + as.numeric(as.factor(sex)), data = crabs)
#> 
#> Project-Pursuit Oblique Decision Tree:
#> If ([ 0.59 0.17 0.09 -0.55 0.48 0.29 ] * x) < 1.136017:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 

# Example 6: formula interface with the `crabs` dataset with regularization
PPTree(Type ~ . - sex + as.numeric(as.factor(sex)), data = crabs, lambda = 0.5)
#> 
#> Project-Pursuit Oblique Decision Tree:
#> If ([ 0.71 0.25 0 -0.13 0.65 -0.04 ] * x) < 18.34709:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 

# Example 7: matrix interface with the `crabs` dataset
x <- crabs[, 2:5]
x$sex <- as.numeric(as.factor(crabs$sex))
PPTree(x = x, y = crabs$Type)
#> If ([ 0.67 0.16 0.31 -0.59 0.27 ] * x) < 1.406607:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 

# Example 8: matrix interface with the `crabs` dataset with regularization
x <- crabs[, 2:5]
x$sex <- as.numeric(as.factor(crabs$sex))
PPTree(x = x, y = crabs$Type, lambda = 0.5)
#> If ([ 0.89 0.41 0.06 -0.08 0.17 ] * x) < 18.43002:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
```
