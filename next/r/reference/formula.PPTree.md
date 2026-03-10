# Extracts the formula used to train a PPTree model.

Extracts the formula used to train a PPTree model.

## Usage

``` r
# S3 method for class 'PPTree'
formula(x, ...)
```

## Arguments

- x:

  A PPTree model.

- ...:

  (unused) other parameters typically passed to formula.

## Value

The formula used to train the model.

## See also

[`PPTree`](https://andres-vidal.github.io/pptree/main/r/reference/PPTree.md)
for training,
[`predict.PPTree`](https://andres-vidal.github.io/pptree/main/r/reference/predict.PPTree.md),
[`summary.PPTree`](https://andres-vidal.github.io/pptree/main/r/reference/summary.PPTree.md)

## Examples

``` r
model <- PPTree(Type ~ ., data = iris)
formula(model)
#> Type ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 
#>     1
#> <environment: 0x55ef89662778>
```
