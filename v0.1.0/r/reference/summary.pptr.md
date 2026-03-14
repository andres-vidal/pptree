# Summarizes a pptr model.

Summarizes a pptr model.

## Usage

``` r
# S3 method for class 'pptr'
summary(object, ...)
```

## Arguments

- object:

  A pptr model.

- ...:

  (unused) other parameters typically passed to summary.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pptr.md),
[`predict.pptr`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/predict.pptr.md),
[`print.pptr`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/print.pptr.md)

## Examples

``` r
model <- pptr(Type ~ ., data = iris)
summary(model)
#> 
#> Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> 150 observations of 4 features
#> Regularization parameter: 0 
#> Classes:
#>  setosa
#>  versicolor
#>  virginica 
#> Formula:
#>  Type ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width -      1 
#> -------------------------------------
#> Variable Importance:
#> 
#>       Variable         σ Projection
#> 1 Petal.Length 1.7652982  0.7787341
#> 2  Petal.Width 0.7622377  0.3437867
#> 3  Sepal.Width 0.4358663  0.1806545
#> 4 Sepal.Length 0.8280661  0.1471821
#> 
#> Note: Variable importance was calculated using scaled coefficients (|a_j| * σ_j).
#> Variable contributions can only be theoretically interpreted as such
#> if the model was trained on scaled data. Scaling also changes the
#> projection-pursuit optimization, which may affect the resulting tree.
#> -------------------------------------
#> Confusion Matrix:
#> 
#>             Predicted
#> Actual       setosa versicolor virginica
#>   setosa         50          0         0
#>   versicolor      0         48         2
#>   virginica       0          1        49
#> 
#> Training error: 2 %
#> 
```
