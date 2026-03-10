# Trains a Random Forest of Project-Pursuit oblique decision trees.

This function trains a Random Forest of Project-Pursuit oblique decision
tree (PPTree) using either a formula and data frame interface or a
matrix-based interface. When using the formula interface, specify the
model formula and the data frame containing the variables. For the
matrix-based interface, provide matrices for the features and labels
directly. The number of trees is controlled by the `size` parameter.
Each tree is trained on a stratified bootstrap sample drawn from the
data. The number of variables to consider at each split is controlled by
the `n_vars` parameter. If `lambda = 0`, the model is trained using
Linear Discriminant Analysis (LDA). If `lambda > 0`, the model is
trained using Penalized Discriminant Analysis (PDA).

## Usage

``` r
PPForest(
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

A PPForest model trained on `x` and `y`.

## See also

[`predict.PPForest`](https://andres-vidal.github.io/pptree/main/r/reference/predict.PPForest.md),
[`formula.PPForest`](https://andres-vidal.github.io/pptree/main/r/reference/formula.PPForest.md),
[`summary.PPForest`](https://andres-vidal.github.io/pptree/main/r/reference/summary.PPForest.md),
[`print.PPForest`](https://andres-vidal.github.io/pptree/main/r/reference/print.PPForest.md),
[`pp_forest`](https://andres-vidal.github.io/pptree/main/r/reference/pp_forest.md)
for parsnip integration

## Examples

``` r
# Example 1: formula interface with the `iris` dataset
PPForest(Type ~ ., data = iris)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.01 -0.82 0.53 0.2 ] * x) < -0.6616913:
#>   Predict: setosa 
#> Else:
#>  If ([ 0.09 0.48 -0.34 -0.8 ] * x) < -1.103081:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> 
#> Tree 2:
#> If ([ 0.34 0.63 -0.68 -0.15 ] * x) < 1.586351:
#>  If ([ 0.15 0.38 -0.34 -0.85 ] * x) < -0.9924551:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
#> 

# Example 2: formula interface with the `iris` dataset with regularization
PPForest(Type ~ ., data = iris, lambda = 0.5)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.08 -0.72 0.43 0.53 ] * x) < 0.03756618:
#>   Predict: setosa 
#> Else:
#>  If ([ 0.05 0.11 -0.31 -0.94 ] * x) < -2.512249:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> 
#> Tree 2:
#> If ([ 0.11 -0.63 0.55 0.53 ] * x) < 0.9280792:
#>   Predict: setosa 
#> Else:
#>  If ([ 0.01 -0.22 0.36 0.91 ] * x) < 2.788582:
#>    Predict: versicolor 
#>  Else:
#>    Predict: virginica 
#> 
#> 

# Example 3: matrix interface with the `iris` dataset
PPForest(x = iris[, 1:4], y = iris[, 5])
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

# Example 4: matrix interface with the `iris` dataset with regularization
PPForest(x = iris[, 1:4], y = iris[, 5], lambda = 0.5)
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

# Example 5: formula interface with the `crabs` dataset
PPForest(Type ~ . - sex + as.numeric(as.factor(sex)), data = crabs)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.64 0.06 -0.05 -0.47 0.59 0.13 ] * x) < 0.6518859:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> Tree 2:
#> If ([ 0.65 0.13 0.13 -0.58 0.42 0.17 ] * x) < 1.011468:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> 

# Example 6: formula interface with the `crabs` dataset with regularization
PPForest(Type ~ . - sex + as.numeric(as.factor(sex)), data = crabs, lambda = 0.5)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.39 0.18 0 -0.07 0.37 -0.82 ] * x) < 9.471808:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> Tree 2:
#> If ([ 0.55 0.12 0.01 -0.09 0.51 0.64 ] * x) < 15.25045:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> 

# Example 7: matrix interface with the `crabs` dataset
x <- crabs[, 2:5]
x$sex <- as.numeric(as.factor(crabs$sex))
PPForest(x = x, y = crabs$Type)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.64 0.24 0.22 -0.53 0.45 ] * x) < 1.742541:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> Tree 2:
#> If ([ 0.75 0.16 0.25 -0.56 0.19 ] * x) < 1.37413:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> 

# Example 8: matrix interface with the `crabs` dataset with regularization
x <- crabs[, 2:5]
x$sex <- as.numeric(as.factor(crabs$sex))
PPForest(x = x, y = crabs$Type, lambda = 0.5)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.82 0.41 0.07 -0.05 -0.39 ] * x) < 18.13947:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> Tree 2:
#> If ([ 0.91 0.35 0.06 -0.08 -0.2 ] * x) < 17.40269:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> 
```
