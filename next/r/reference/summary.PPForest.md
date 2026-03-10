# Summarizes a PPForest model.

Summarizes a PPForest model.

## Usage

``` r
# S3 method for class 'PPForest'
summary(object, ...)
```

## Arguments

- object:

  A PPForest model.

- ...:

  (unused) other parameters typically passed to summary.

## See also

[`PPForest`](https://andres-vidal.github.io/pptree/main/r/reference/PPForest.md),
[`predict.PPForest`](https://andres-vidal.github.io/pptree/main/r/reference/predict.PPForest.md),
[`print.PPForest`](https://andres-vidal.github.io/pptree/main/r/reference/print.PPForest.md)

## Examples

``` r
model <- PPForest(Type ~ ., data = iris)
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
#>       Variable         σ Projection  Weighted   Permuted
#> 1 Petal.Length 1.7652982  0.6765911 0.7033836 0.47511965
#> 2  Petal.Width 0.7622377  0.3702932 0.3385268 0.24130782
#> 3  Sepal.Width 0.4358663  0.1965394 0.2130345 0.06331739
#> 4 Sepal.Length 0.8280661  0.1442678 0.1379393 0.04545453
#> -------------------------------------
#> Confusion Matrix:
#> TODO
```
