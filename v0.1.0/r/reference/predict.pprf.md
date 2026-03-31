# Predicts the labels or vote proportions of a set of observations using a pprf model.

Predicts the labels or vote proportions of a set of observations using a
pprf model.

## Usage

``` r
# S3 method for class 'pprf'
predict(object, new_data = NULL, type = "class", ...)
```

## Arguments

- object:

  A pprf model.

- new_data:

  A data frame or matrix of new observations to predict. If `NULL`, the
  first positional argument in `...` is used for backward compatibility.

- type:

  The type of prediction: `"class"` (default) returns a factor of
  predicted labels, `"prob"` returns a data frame of vote proportions.

- ...:

  For backward compatibility, the first positional argument is treated
  as `new_data` when `new_data` is `NULL`.

## Value

If `type = "class"`, a factor of predicted labels. If `type = "prob"`, a
data frame with one column per group, where each row sums to 1.

## See also

[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pprf.md)
for training,
[`formula.pprf`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/formula.pprf.md),
[`summary.pprf`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/summary.pprf.md)

## Examples

``` r
# Example 1: with the `iris` dataset
model <- pprf(Type ~ ., data = iris)
predict(model, iris)
#>   [1] setosa     setosa     setosa     setosa     setosa     setosa    
#>   [7] setosa     setosa     setosa     setosa     setosa     setosa    
#>  [13] setosa     setosa     setosa     setosa     setosa     setosa    
#>  [19] setosa     setosa     setosa     setosa     setosa     setosa    
#>  [25] setosa     setosa     setosa     setosa     setosa     setosa    
#>  [31] setosa     setosa     setosa     setosa     setosa     setosa    
#>  [37] setosa     setosa     setosa     setosa     setosa     setosa    
#>  [43] setosa     setosa     setosa     setosa     setosa     setosa    
#>  [49] setosa     setosa     versicolor versicolor versicolor versicolor
#>  [55] versicolor versicolor versicolor versicolor versicolor versicolor
#>  [61] versicolor versicolor versicolor versicolor versicolor versicolor
#>  [67] versicolor versicolor versicolor versicolor virginica  versicolor
#>  [73] versicolor versicolor versicolor versicolor versicolor versicolor
#>  [79] versicolor versicolor versicolor versicolor versicolor virginica 
#>  [85] versicolor versicolor versicolor versicolor versicolor versicolor
#>  [91] versicolor versicolor versicolor versicolor versicolor versicolor
#>  [97] versicolor versicolor versicolor versicolor virginica  virginica 
#> [103] virginica  virginica  virginica  virginica  virginica  virginica 
#> [109] virginica  virginica  virginica  virginica  virginica  virginica 
#> [115] virginica  virginica  virginica  virginica  virginica  virginica 
#> [121] virginica  virginica  virginica  virginica  virginica  virginica 
#> [127] virginica  virginica  virginica  virginica  virginica  virginica 
#> [133] virginica  versicolor virginica  virginica  virginica  virginica 
#> [139] virginica  virginica  virginica  virginica  virginica  virginica 
#> [145] virginica  virginica  virginica  virginica  virginica  virginica 
#> Levels: setosa versicolor virginica

# Example 2: with the `crabs` dataset
model <- pprf(Type ~ ., data = crabs)
predict(model, crabs)
#>   [1] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
#>  [38] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
#>  [75] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
#> [112] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
#> [149] B B O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O
#> [186] O O O O O O O O O O O O O O O
#> Levels: B O

# Example 3: vote proportions
model <- pprf(Type ~ ., data = iris)
predict(model, iris, type = "prob")
#>     setosa versicolor virginica
#> 1        1        0.0       0.0
#> 2        1        0.0       0.0
#> 3        1        0.0       0.0
#> 4        1        0.0       0.0
#> 5        1        0.0       0.0
#> 6        1        0.0       0.0
#> 7        1        0.0       0.0
#> 8        1        0.0       0.0
#> 9        1        0.0       0.0
#> 10       1        0.0       0.0
#> 11       1        0.0       0.0
#> 12       1        0.0       0.0
#> 13       1        0.0       0.0
#> 14       1        0.0       0.0
#> 15       1        0.0       0.0
#> 16       1        0.0       0.0
#> 17       1        0.0       0.0
#> 18       1        0.0       0.0
#> 19       1        0.0       0.0
#> 20       1        0.0       0.0
#> 21       1        0.0       0.0
#> 22       1        0.0       0.0
#> 23       1        0.0       0.0
#> 24       1        0.0       0.0
#> 25       1        0.0       0.0
#> 26       1        0.0       0.0
#> 27       1        0.0       0.0
#> 28       1        0.0       0.0
#> 29       1        0.0       0.0
#> 30       1        0.0       0.0
#> 31       1        0.0       0.0
#> 32       1        0.0       0.0
#> 33       1        0.0       0.0
#> 34       1        0.0       0.0
#> 35       1        0.0       0.0
#> 36       1        0.0       0.0
#> 37       1        0.0       0.0
#> 38       1        0.0       0.0
#> 39       1        0.0       0.0
#> 40       1        0.0       0.0
#> 41       1        0.0       0.0
#> 42       1        0.0       0.0
#> 43       1        0.0       0.0
#> 44       1        0.0       0.0
#> 45       1        0.0       0.0
#> 46       1        0.0       0.0
#> 47       1        0.0       0.0
#> 48       1        0.0       0.0
#> 49       1        0.0       0.0
#> 50       1        0.0       0.0
#> 51       0        1.0       0.0
#> 52       0        1.0       0.0
#> 53       0        1.0       0.0
#> 54       0        1.0       0.0
#> 55       0        1.0       0.0
#> 56       0        1.0       0.0
#> 57       0        1.0       0.0
#> 58       0        1.0       0.0
#> 59       0        1.0       0.0
#> 60       0        1.0       0.0
#> 61       0        1.0       0.0
#> 62       0        1.0       0.0
#> 63       0        1.0       0.0
#> 64       0        1.0       0.0
#> 65       0        1.0       0.0
#> 66       0        1.0       0.0
#> 67       0        1.0       0.0
#> 68       0        1.0       0.0
#> 69       0        1.0       0.0
#> 70       0        1.0       0.0
#> 71       0        0.5       0.5
#> 72       0        1.0       0.0
#> 73       0        0.5       0.5
#> 74       0        1.0       0.0
#> 75       0        1.0       0.0
#> 76       0        1.0       0.0
#> 77       0        1.0       0.0
#> 78       0        1.0       0.0
#> 79       0        1.0       0.0
#> 80       0        1.0       0.0
#> 81       0        1.0       0.0
#> 82       0        1.0       0.0
#> 83       0        1.0       0.0
#> 84       0        0.0       1.0
#> 85       0        1.0       0.0
#> 86       0        1.0       0.0
#> 87       0        1.0       0.0
#> 88       0        1.0       0.0
#> 89       0        1.0       0.0
#> 90       0        1.0       0.0
#> 91       0        1.0       0.0
#> 92       0        1.0       0.0
#> 93       0        1.0       0.0
#> 94       0        1.0       0.0
#> 95       0        1.0       0.0
#> 96       0        1.0       0.0
#> 97       0        1.0       0.0
#> 98       0        1.0       0.0
#> 99       0        1.0       0.0
#> 100      0        1.0       0.0
#> 101      0        0.0       1.0
#> 102      0        0.0       1.0
#> 103      0        0.0       1.0
#> 104      0        0.0       1.0
#> 105      0        0.0       1.0
#> 106      0        0.0       1.0
#> 107      0        0.0       1.0
#> 108      0        0.0       1.0
#> 109      0        0.0       1.0
#> 110      0        0.0       1.0
#> 111      0        0.0       1.0
#> 112      0        0.0       1.0
#> 113      0        0.0       1.0
#> 114      0        0.0       1.0
#> 115      0        0.0       1.0
#> 116      0        0.0       1.0
#> 117      0        0.0       1.0
#> 118      0        0.0       1.0
#> 119      0        0.0       1.0
#> 120      0        0.0       1.0
#> 121      0        0.0       1.0
#> 122      0        0.0       1.0
#> 123      0        0.0       1.0
#> 124      0        0.0       1.0
#> 125      0        0.0       1.0
#> 126      0        0.0       1.0
#> 127      0        0.0       1.0
#> 128      0        0.0       1.0
#> 129      0        0.0       1.0
#> 130      0        0.0       1.0
#> 131      0        0.0       1.0
#> 132      0        0.0       1.0
#> 133      0        0.0       1.0
#> 134      0        0.5       0.5
#> 135      0        0.0       1.0
#> 136      0        0.0       1.0
#> 137      0        0.0       1.0
#> 138      0        0.0       1.0
#> 139      0        0.0       1.0
#> 140      0        0.0       1.0
#> 141      0        0.0       1.0
#> 142      0        0.0       1.0
#> 143      0        0.0       1.0
#> 144      0        0.0       1.0
#> 145      0        0.0       1.0
#> 146      0        0.0       1.0
#> 147      0        0.0       1.0
#> 148      0        0.0       1.0
#> 149      0        0.0       1.0
#> 150      0        0.0       1.0
```
