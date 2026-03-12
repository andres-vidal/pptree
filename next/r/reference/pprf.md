# Trains a Random Forest of Project-Pursuit oblique decision trees.

This function trains a Random Forest of Project-Pursuit oblique decision
tree using either a formula and data frame interface or a matrix-based
interface. When using the formula interface, specify the model formula
and the data frame containing the variables. For the matrix-based
interface, provide matrices for the features and labels directly. The
number of trees is controlled by the `size` parameter. Each tree is
trained on a stratified bootstrap sample drawn from the data. The number
of variables to consider at each split is controlled by the `n_vars`
parameter. If `lambda = 0`, the model is trained using Linear
Discriminant Analysis (LDA). If `lambda > 0`, the model is trained using
Penalized Discriminant Analysis (PDA).

## Usage

``` r
pprf(
  formula = NULL,
  data = NULL,
  x = NULL,
  y = NULL,
  size = 2,
  lambda = 0,
  n_vars = NULL,
  seed = NULL,
  n_threads = NULL
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

- size:

  The number of trees in the forest.

- lambda:

  A regularization parameter. If `lambda = 0`, the model is trained
  using Linear Discriminant Analysis (LDA). If `lambda > 0`, the model
  is trained using Penalized Discriminant Analysis (PDA).

- n_vars:

  The number of variables to consider at each split. These are chosen
  uniformly in each split. The default is all variables.

- seed:

  An optional integer seed for reproducibility. If `NULL` (default), a
  seed is drawn from R's RNG, so
  [`set.seed()`](https://rdrr.io/r/base/Random.html) controls
  reproducibility. If an integer is provided, that value is used
  directly. The same seed is used for training and for computing
  permuted variable importance.

- n_threads:

  The number of threads to use. The default is the number of cores
  available.

## Value

A pprf model trained on `x` and `y`.

## See also

[`predict.pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/predict.pprf.md),
[`formula.pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/formula.pprf.md),
[`summary.pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/summary.pprf.md),
[`print.pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/print.pprf.md),
[`pp_rand_forest`](https://andres-vidal.github.io/ppforest2/main/r/reference/pp_rand_forest.md)
for parsnip integration

## Examples

``` r
# Example 1: formula interface with the `iris` dataset
pprf(Type ~ ., data = iris)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.27 0.63 -0.69 -0.21 ] * x) < 1.063238:
#>  If ([ 0.31 0.08 -0.48 -0.82 ] * x) < -1.619433:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
#> Tree 2:
#> If ([ 0.23 0.72 -0.65 -0.03 ] * x) < 1.453875:
#>  If ([ 0.21 0.49 -0.55 -0.64 ] * x) < -1.026834:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
#> 

# Example 2: formula interface with the `iris` dataset with regularization
pprf(Type ~ ., data = iris, lambda = 0.5)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.08 -0.67 0.43 0.6 ] * x) < 0.2777239:
#>   Predict: setosa 
#> Else:
#>  If ([ 0.02 0.05 -0.24 -0.97 ] * x) < -2.578496:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> 
#> Tree 2:
#> If ([ 0.09 -0.68 0.46 0.56 ] * x) < 0.3777013:
#>   Predict: setosa 
#> Else:
#>  If ([ 0.01 -0.15 0.33 0.93 ] * x) < 2.815652:
#>    Predict: versicolor 
#>  Else:
#>    Predict: virginica 
#> 
#> 

# Example 3: matrix interface with the `iris` dataset
pprf(x = iris[, 1:4], y = iris[, 5])
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.17 0.76 -0.56 -0.29 ] * x) < 1.249054:
#>  If ([ 0.19 0.33 -0.31 -0.87 ] * x) < -0.8064449:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
#> Tree 2:
#> If ([ 0.05 -0.79 0.59 0.17 ] * x) < -0.2461427:
#>   Predict: setosa 
#> Else:
#>  If ([ 0.12 0.42 -0.27 -0.86 ] * x) < -0.8229396:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> 
#> 

# Example 4: matrix interface with the `iris` dataset with regularization
pprf(x = iris[, 1:4], y = iris[, 5], lambda = 0.5)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.08 -0.71 0.47 0.51 ] * x) < 0.1726278:
#>   Predict: setosa 
#> Else:
#>  If ([ 0.01 0.23 -0.47 -0.85 ] * x) < -3.015483:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> 
#> Tree 2:
#> If ([ 0.11 -0.72 0.43 0.53 ] * x) < 0.2246059:
#>   Predict: setosa 
#> Else:
#>  If ([ 0.07 0.19 -0.38 -0.91 ] * x) < -2.423834:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> 
#> 

# Example 5: formula interface with the `crabs` dataset
pprf(Type ~ . - sex + as.numeric(as.factor(sex)), data = crabs)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.5 0.27 0 -0.45 0.43 0.53 ] * x) < 1.608614:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> Tree 2:
#> If ([ 0.63 0.18 0.04 -0.53 0.49 0.22 ] * x) < 1.145166:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> 

# Example 6: formula interface with the `crabs` dataset with regularization
pprf(Type ~ . - sex + as.numeric(as.factor(sex)), data = crabs, lambda = 0.5)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.61 0.25 0.01 -0.09 0.55 -0.5 ] * x) < 16.8111:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> Tree 2:
#> If ([ 0.73 0.21 0 -0.12 0.56 -0.31 ] * x) < 17.04963:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> 

# Example 7: matrix interface with the `crabs` dataset
x <- crabs[, 2:5]
x$sex <- as.numeric(as.factor(crabs$sex))
pprf(x = x, y = crabs$Type)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.63 0.25 0.25 -0.55 0.42 ] * x) < 1.573025:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> Tree 2:
#> If ([ 0.72 0.18 0.24 -0.56 0.28 ] * x) < 1.340618:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> 

# Example 8: matrix interface with the `crabs` dataset with regularization
x <- crabs[, 2:5]
x$sex <- as.numeric(as.factor(crabs$sex))
pprf(x = x, y = crabs$Type, lambda = 0.5)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.81 0.38 0.04 -0.09 0.43 ] * x) < 16.06476:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> Tree 2:
#> If ([ 0.53 0.2 0.03 -0.06 0.83 ] * x) < 10.87641:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> 
```
