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

[`pptr`](https://andres-vidal.github.io/ppforest2/main/r/reference/pptr.md),
[`predict.pptr`](https://andres-vidal.github.io/ppforest2/main/r/reference/predict.pptr.md),
[`print.pptr`](https://andres-vidal.github.io/ppforest2/main/r/reference/print.pptr.md)

## Examples

``` r
model <- pptr(Type ~ ., data = iris)
summary(model)
#> 
#> Project-Pursuit Oblique Decision Tree
#> 
#> Regularization parameter: 0 
#> 
#> Data Summary:
#>   observations: 150 
#>   features:     4 
#>   groups:       3 
#>   group names:  setosa, versicolor, virginica 
#>   formula:      Type ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width -      1 
#> 
#> Confusion Matrix:
#> 
#>             Predicted
#> Actual       setosa versicolor virginica
#>   setosa         50          0         0
#>   versicolor      0         48         2
#>   virginica       0          1        49
#> 
#> Training error: 2%
#> 
#> Variable Importance:
#> 
#>       Variable         σ Projection
#> 1 Petal.Length 1.7652982 0.10064526
#> 2  Petal.Width 0.7622377 0.06144172
#> 3 Sepal.Length 0.8280661 0.02164980
#> 4  Sepal.Width 0.4358663 0.02155388
#> 
#> Note: Variable importance was calculated using scaled coefficients (|a_j| * σ_j).
#> Variable contributions can only be theoretically interpreted as such
#> if the model was trained on scaled data. Scaling also changes the
#> projection-pursuit optimization, which may affect the resulting tree.
#> 
```
