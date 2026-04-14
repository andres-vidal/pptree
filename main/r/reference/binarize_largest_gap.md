# Largest-gap binarization strategy.

Creates a binarization strategy that reduces multiclass nodes to binary
by projecting group means and splitting at the largest gap. This is the
default binarization strategy.

## Usage

``` r
binarize_largest_gap()
```

## Value

A `binarize_strategy` object.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/main/r/reference/pptr.md),
[`pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/pprf.md)

## Examples

``` r
binarize_largest_gap()
#> $name
#> [1] "largest_gap"
#> 
#> $display_name
#> [1] "Largest gap"
#> 
#> attr(,"class")
#> [1] "binarize_strategy"
```
