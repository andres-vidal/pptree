# Extracts the formula used to train a PPForest model.

Extracts the formula used to train a PPForest model.

## Usage

``` r
# S3 method for class 'PPForest'
formula(x, ...)
```

## Arguments

- x:

  A PPForest model.

- ...:

  (unused) other parameters typically passed to formula.

## Value

The formula used to train the model.

## See also

[`PPForest`](https://andres-vidal.github.io/pptree/main/r/reference/PPForest.md)
for training,
[`predict.PPForest`](https://andres-vidal.github.io/pptree/main/r/reference/predict.PPForest.md),
[`summary.PPForest`](https://andres-vidal.github.io/pptree/main/r/reference/summary.PPForest.md)

## Examples

``` r
model <- PPForest(Type ~ ., data = iris)
formula(model)
#> Type ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 
#>     1
#> <environment: 0x55e240189818>
```
