# Prints a PPForest model.

Prints a PPForest model.

## Usage

``` r
# S3 method for class 'PPForest'
print(x, ...)
```

## Arguments

- x:

  A PPForest model.

- ...:

  (unused) other parameters typically passed to print.

## See also

[`PPForest`](https://andres-vidal.github.io/pptree/main/r/reference/PPForest.md),
[`summary.PPForest`](https://andres-vidal.github.io/pptree/main/r/reference/summary.PPForest.md)

## Examples

``` r
model <- PPForest(Type ~ ., data = iris)
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
