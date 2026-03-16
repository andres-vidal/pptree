# Prints a pprf model.

Prints a pprf model.

## Usage

``` r
# S3 method for class 'pprf'
print(x, ...)
```

## Arguments

- x:

  A pprf model.

- ...:

  (unused) other parameters typically passed to print.

## See also

[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pprf.md),
[`summary.pprf`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/summary.pprf.md)

## Examples

``` r
model <- pprf(Type ~ ., data = iris)
print(model)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0 0.05 -0.03 -0.02 ] * x) < 0.03828842:
#>  If ([ 0.01 0.07 -0.05 -0.21 ] * x) < -0.3441847:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
#> Tree 2:
#> If ([ 0.02 0.04 -0.04 -0.01 ] * x) < 0.0853334:
#>  If ([ 0.05 0.09 -0.07 -0.17 ] * x) < -0.09787051:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
#> 
```
