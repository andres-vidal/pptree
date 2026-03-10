# Summarizes a PPTree model.

Summarizes a PPTree model.

## Usage

``` r
# S3 method for class 'PPTree'
summary(object, ...)
```

## Arguments

- object:

  A PPTree model.

- ...:

  (unused) other parameters typically passed to summary.

## See also

[`PPTree`](https://andres-vidal.github.io/pptree/main/r/reference/PPTree.md),
[`predict.PPTree`](https://andres-vidal.github.io/pptree/main/r/reference/predict.PPTree.md),
[`print.PPTree`](https://andres-vidal.github.io/pptree/main/r/reference/print.PPTree.md)

## Examples

``` r
model <- PPTree(Type ~ ., data = iris)
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
#> -------------------------------------
#> Confusion Matrix:
#> TODO
```
