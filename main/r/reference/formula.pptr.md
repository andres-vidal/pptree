# Extracts the formula used to train a pptr model.

Extracts the formula used to train a pptr model.

## Usage

``` r
# S3 method for class 'pptr'
formula(x, ...)
```

## Arguments

- x:

  A pptr model.

- ...:

  (unused) other parameters typically passed to formula.

## Value

The formula used to train the model.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/main/r/reference/pptr.md)
for training,
[`predict.pptr`](https://andres-vidal.github.io/ppforest2/main/r/reference/predict.pptr.md),
[`summary.pptr`](https://andres-vidal.github.io/ppforest2/main/r/reference/summary.pptr.md)

## Examples

``` r
model <- pptr(Type ~ ., data = iris)
formula(model)
#> Type ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 
#>     1
#> <environment: 0x558ce225b218>
```
