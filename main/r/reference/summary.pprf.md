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
#> 
#> Size: 2 trees
#> pp method: LDA (lambda=0)
#> dr method: Uniform random (n_vars=4)
#> sr method: Mean of means
#> 
#> 
#> Data Summary:
#>   observations: 150 
#>   features:     4 
#>   groups:       3 
#>   group names:  setosa, versicolor, virginica 
#>   formula:      Type ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width -      1 
#> 
#> Training Confusion Matrix:
#> 
#>             Predicted
#> Actual       setosa versicolor virginica
#>   setosa         50          0         0
#>   versicolor      0         49         1
#>   virginica       0          1        49
#> 
#> Training error: 1.33%
#> 
#> OOB Confusion Matrix:
#> 
#>             Predicted
#> Actual       setosa versicolor virginica
#>   setosa         30          0         0
#>   versicolor      0         29         0
#>   virginica       0          0        31
#> 
#> OOB error: 0%
#> 
#> Variable Importance:
#> 
#>       Variable         σ Projection   Weighted   Permuted
#> 1 Petal.Length 1.7652982 0.11812986 0.10914339 0.66332722
#> 2  Petal.Width 0.7622377 0.04823423 0.03898794 0.22586522
#> 3 Sepal.Length 0.8280661 0.02621456 0.02295329 0.05950212
#> 4  Sepal.Width 0.4358663 0.01906149 0.01900462 0.06375226
#> 
#> Note: Variable importance was calculated using scaled coefficients (|a_j| * σ_j).
#> Variable contributions can only be theoretically interpreted as such
#> if the model was trained on scaled data. Scaling also changes the
#> projection-pursuit optimization, which may affect the resulting tree.
#> 
```
