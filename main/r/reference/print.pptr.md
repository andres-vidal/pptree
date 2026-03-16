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

[`pptr`](https://andres-vidal.github.io/ppforest2/main/r/reference/pptr.md),
[`summary.pptr`](https://andres-vidal.github.io/ppforest2/main/r/reference/summary.pptr.md)

## Examples

``` r
model <- pptr(Type ~ ., data = iris)
print(model)
#> 
#> Project-Pursuit Oblique Decision Tree:
#> If ([ 0.01 0.04 -0.04 -0.01 ] * x) < 0.06660754:
#>  If ([ 0.04 0.07 -0.09 -0.15 ] * x) < -0.2075133:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
```
