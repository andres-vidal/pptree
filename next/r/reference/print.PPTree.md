# Prints a PPTree model.

Prints a PPTree model.

## Usage

``` r
# S3 method for class 'PPTree'
print(x, ...)
```

## Arguments

- x:

  A PPTree model.

- ...:

  (unused) other parameters typically passed to print.

## See also

[`PPTree`](https://andres-vidal.github.io/pptree/main/r/reference/PPTree.md),
[`summary.PPTree`](https://andres-vidal.github.io/pptree/main/r/reference/summary.PPTree.md)

## Examples

``` r
model <- PPTree(Type ~ ., data = iris)
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
