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
#> If ([ 0.01 0.04 -0.04 -0.02 ] * x) < 0.07012913:
#>  If ([ 0.05 0.06 -0.11 -0.13 ] * x) < -0.2990253:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
#> Tree 2:
#> If ([ 0.01 0.04 -0.04 -0.01 ] * x) < 0.07433026:
#>  If ([ 0.03 0.03 -0.06 -0.16 ] * x) < -0.2947403:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
#> 
```
