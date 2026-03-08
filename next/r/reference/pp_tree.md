# Parsnip model specification for PPTree.

Creates a model specification for a single Projection Pursuit decision
tree. Use `set_engine("PPTree")` to select the PPTree engine.

## Usage

``` r
pp_tree(mode = "classification", penalty = NULL)
```

## Arguments

- mode:

  A character string for the model type. Only `"classification"` is
  supported.

- penalty:

  The regularization parameter (maps to `lambda`).

## Value

A parsnip model specification.

## See also

[`PPTree`](https://andres-vidal.github.io/pptree/main/r/reference/PPTree.md)
for the underlying training function,
[`pp_forest`](https://andres-vidal.github.io/pptree/main/r/reference/pp_forest.md)
for forests

## Examples

``` r
if (FALSE) { # \dontrun{
library(parsnip)
spec <- pp_tree(penalty = 0) %>% set_engine("PPTree")
fit <- spec %>% fit(Type ~ ., data = iris)
predict(fit, iris)
} # }
```
