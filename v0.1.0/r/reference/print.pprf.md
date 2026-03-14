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
#> If ([ 0.04 0.78 -0.53 -0.32 ] * x) < 0.6079259:
#>  If ([ 0.05 0.31 -0.24 -0.92 ] * x) < -1.518452:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
#> Tree 2:
#> If ([ 0.26 0.68 -0.67 -0.16 ] * x) < 1.333983:
#>  If ([ 0.23 0.41 -0.34 -0.81 ] * x) < -0.4588342:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
#> 
```
