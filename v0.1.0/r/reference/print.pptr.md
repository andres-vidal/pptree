# Prints a pptr model.

Prints a pptr model.

## Usage

``` r
# S3 method for class 'pptr'
print(x, ...)
```

## Arguments

- x:

  A pptr model.

- ...:

  (unused) other parameters typically passed to print.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pptr.md),
[`summary.pptr`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/summary.pptr.md)

## Examples

``` r
model <- pptr(Type ~ ., data = iris)
print(model)
#> 
#> Project-Pursuit Oblique Decision Tree:
#> If ([ 0.19 0.71 -0.66 -0.17 ] * x) < 1.070804:
#>  If ([ 0.23 0.36 -0.44 -0.79 ] * x) < -1.062904:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
```
