# Parsnip model specification for PPForest.

Creates a model specification for a Projection Pursuit random forest.
Use `set_engine("PPTree")` to select the PPTree engine.

## Usage

``` r
pp_forest(mode = "classification", trees = NULL, mtry = NULL, penalty = NULL)
```

## Arguments

- mode:

  A character string for the model type. Only `"classification"` is
  supported.

- trees:

  The number of trees in the forest (maps to `size`).

- mtry:

  The number of variables to consider at each split (maps to `n_vars`).

- penalty:

  The regularization parameter (maps to `lambda`).

## Value

A parsnip model specification.

## See also

[`PPForest`](https://andres-vidal.github.io/pptree/main/r/reference/PPForest.md)
for the underlying training function,
[`pp_tree`](https://andres-vidal.github.io/pptree/main/r/reference/pp_tree.md)
for single trees

## Examples

``` r
if (FALSE) { # \dontrun{
library(parsnip)
spec <- pp_forest(trees = 50, mtry = 2) %>% set_engine("PPTree")
fit <- spec %>% fit(Type ~ ., data = iris)
predict(fit, iris)
predict(fit, iris, type = "prob")
} # }
```
