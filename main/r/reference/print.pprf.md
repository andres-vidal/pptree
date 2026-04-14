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

[`pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/pprf.md),
[`summary.pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/summary.pprf.md)

## Examples

``` r
model <- pprf(Type ~ ., data = iris)
print(model)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.01 0.05 -0.04 -0.01 ] * x) < 0.0902002:
#>  If ([ 0.04 0.08 -0.07 -0.18 ] * x) < -0.1816266:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
#> Tree 2:
#> If ([ 0.03 0.03 -0.05 -0.01 ] * x) < 0.1153414:
#>  If ([ 0.07 0.06 -0.1 -0.17 ] * x) < -0.1376903:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
#> 
```
