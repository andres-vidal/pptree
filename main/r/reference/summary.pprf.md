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
#> vars method: Uniform random (count=4)
#> cutpoint method: Mean of means
#> stop rule: Pure node
#> binarize method: Largest gap
#> partition method: By group
#> leaf method: Majority vote
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
#>   virginica       0          4        46
#> 
#> Training error: 3.33%
#> 
#> OOB Confusion Matrix:
#> 
#>             Predicted
#> Actual       setosa versicolor virginica
#>   setosa         34          0         0
#>   versicolor      0         31         0
#>   virginica       0          3        30
#> 
#> OOB error: 3.06%
#> 
#> Variable Importance:
#> 
#>       Variable         σ Projection   Weighted   Permuted
#> 1 Petal.Length 1.7652982 0.07846245 0.07242692 0.41032788
#> 2  Petal.Width 0.7622377 0.07662326 0.06041355 0.30295083
#> 3  Sepal.Width 0.4358663 0.02399131 0.02208606 0.06278688
#> 4 Sepal.Length 0.8280661 0.01523169 0.01314180 0.03000000
#> 
#> Note: Variable importance was calculated using scaled coefficients (|a_j| * σ_j).
#> Variable contributions can only be theoretically interpreted as such
#> if the model was trained on scaled data. Scaling also changes the
#> projection-pursuit optimization, which may affect the resulting tree.
#> 
```
