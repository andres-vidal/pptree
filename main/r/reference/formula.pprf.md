# Extracts the formula used to train a pprf model.

Extracts the formula used to train a pprf model.

## Usage

``` r
# S3 method for class 'pprf'
formula(x, ...)
```

## Arguments

- x:

  A pprf model.

- ...:

  (unused) other parameters typically passed to formula.

## Value

The formula used to train the model.

## See also

[`pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/pprf.md)
for training,
[`predict.pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/predict.pprf.md),
[`summary.pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/summary.pprf.md)

## Examples

``` r
model <- pprf(Type ~ ., data = iris)
formula(model)
#> Type ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 
#>     1
#> <environment: 0x55fa5dd41a78>
```
