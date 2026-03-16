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
  p_vars = NULL,
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

  The number of variables to consider at each split (integer). These are
  chosen uniformly in each split. The default is all variables. Cannot
  be used together with `p_vars`.

- p_vars:

  The proportion of variables to consider at each split (number between
  0 and 1, exclusive). For example, `p_vars = 0.5` uses half the
  features. Cannot be used together with `n_vars`.

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
for parsnip integration,
[`vignette("introduction")`](https://andres-vidal.github.io/ppforest2/main/r/articles/introduction.md)
for a tutorial

## Examples

``` r
# Example 1: formula interface with the `iris` dataset
pprf(Type ~ ., data = iris)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.02 0.04 -0.04 -0.01 ] * x) < 0.06435043:
#>  If ([ 0.06 0.01 -0.09 -0.15 ] * x) < -0.2921683:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
#> Tree 2:
#> If ([ 0.02 0.05 -0.04 0 ] * x) < 0.09979815:
#>  If ([ 0.04 0.09 -0.1 -0.12 ] * x) < -0.189255:
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
#> If ([ 0 -0.04 0.02 0.03 ] * x) < 0.01589731:
#>   Predict: setosa 
#> Else:
#>  If ([ 0 0.01 -0.04 -0.17 ] * x) < -0.4614292:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> 
#> Tree 2:
#> If ([ 0.01 -0.04 0.03 0.03 ] * x) < 0.02173987:
#>   Predict: setosa 
#> Else:
#>  If ([ 0 -0.03 0.06 0.16 ] * x) < 0.4868189:
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
#> If ([ 0.01 0.05 -0.04 -0.02 ] * x) < 0.08010742:
#>  If ([ 0.04 0.07 -0.07 -0.19 ] * x) < -0.1760359:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
#> Tree 2:
#> If ([ 0 -0.05 0.04 0.01 ] * x) < -0.01472723:
#>   Predict: setosa 
#> Else:
#>  If ([ 0.03 0.1 -0.06 -0.2 ] * x) < -0.1889343:
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
#> If ([ 0 -0.04 0.03 0.03 ] * x) < 0.01002465:
#>   Predict: setosa 
#> Else:
#>  If ([ 0 0.04 -0.07 -0.13 ] * x) < -0.4705565:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> 
#> Tree 2:
#> If ([ 0.01 -0.04 0.03 0.03 ] * x) < 0.01345929:
#>   Predict: setosa 
#> Else:
#>  If ([ 0.01 0.03 -0.06 -0.15 ] * x) < -0.3937009:
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
#> If ([ 0.04 0.02 0 -0.04 0.03 0.04 ] * x) < 0.1272064:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> Tree 2:
#> If ([ 0.04 0.01 0 -0.04 0.03 0.01 ] * x) < 0.07738496:
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
#> If ([ 0.01 0 0 0 0.01 -0.01 ] * x) < 0.3201671:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> Tree 2:
#> If ([ 0.01 0 0 0 0.01 -0.01 ] * x) < 0.3204295:
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
#> If ([ 0.06 0.02 0.02 -0.05 0.04 ] * x) < 0.1417684:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> Tree 2:
#> If ([ 0.06 0.01 0.02 -0.04 0.02 ] * x) < 0.1080204:
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
#> If ([ 0.02 0.01 0 0 0.01 ] * x) < 0.3581609:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> Tree 2:
#> If ([ 0.02 0.01 0 0 0.03 ] * x) < 0.3663714:
#>   Predict: B 
#> Else:
#>   Predict: O 
#> 
#> 
```
