# Summarizes a pprf model.

Summarizes a pprf model.

## Usage

``` r
# S3 method for class 'pprf'
summary(object, ...)
```

## Arguments

- object:

  A pprf model.

- ...:

  (unused) other parameters typically passed to summary.

## See also

[`pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/pprf.md),
[`predict.pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/predict.pprf.md),
[`print.pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/print.pprf.md)

## Examples

``` r
model <- pprf(Type ~ ., data = iris)
summary(model)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Size: 2 trees
#> 150 observations of 4 features
#> Regularization parameter: 0 
#> Classes:
#>  setosa
#>  versicolor
#>  virginica 
#> Formula:
#>  Type ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width -      1 
#> OOB error: 2.22 %
#> -------------------------------------
#> Variable Importance:
#> 
#>       Variable         σ Projection   Weighted   Permuted
#> 1 Petal.Length 1.7652982 0.09674555 0.08712038 0.47511965
#> 2  Petal.Width 0.7622377 0.06783789 0.05630573 0.24130782
#> 3  Sepal.Width 0.4358663 0.02741493 0.02527830 0.06331739
#> 4 Sepal.Length 0.8280661 0.02343013 0.02004903 0.04545453
#> 
#> Note: Variable importance was calculated using scaled coefficients (|a_j| * σ_j).
#> Variable contributions can only be theoretically interpreted as such
#> if the model was trained on scaled data. Scaling also changes the
#> projection-pursuit optimization, which may affect the resulting tree.
#> -------------------------------------
#> Confusion Matrix:
#> TODO
```
