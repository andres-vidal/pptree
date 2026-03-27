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
#> Groups:
#>  setosa
#>  versicolor
#>  virginica 
#> Formula:
#>  Type ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width -      1 
#> -------------------------------------
#> Training Confusion Matrix:
#> 
#>             Predicted
#> Actual       setosa versicolor virginica
#>   setosa         50          0         0
#>   versicolor      0         49         1
#>   virginica       0          1        49
#> 
#> Training error: 1.33%
#> -------------------------------------
#> OOB Confusion Matrix:
#> 
#>             Predicted
#> Actual       setosa versicolor virginica
#>   setosa         33          0         0
#>   versicolor      0         27         1
#>   virginica       0          0        31
#> 
#> OOB error: 1.09%
#> -------------------------------------
#> Variable Importance:
#> 
#>       Variable         σ Projection   Weighted   Permuted
#> 1 Petal.Length 1.7652982 0.09914511 0.09107455 0.47878787
#> 2  Petal.Width 0.7622377 0.06957345 0.05445680 0.29393938
#> 3 Sepal.Length 0.8280661 0.02843781 0.02532409 0.03333333
#> 4  Sepal.Width 0.4358663 0.02132891 0.01973198 0.04545456
#> 
#> Note: Variable importance was calculated using scaled coefficients (|a_j| * σ_j).
#> Variable contributions can only be theoretically interpreted as such
#> if the model was trained on scaled data. Scaling also changes the
#> projection-pursuit optimization, which may affect the resulting tree.
#> 
```
