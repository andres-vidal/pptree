# Mean-of-means split rule strategy.

Creates a split rule strategy that splits at the midpoint between group
means. This is the default (and currently only) split rule.

## Usage

``` r
sr_mean_of_means()
```

## Value

A `sr_strategy` object.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/main/r/reference/pptr.md),
[`pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/pprf.md),
[`pp_pda`](https://andres-vidal.github.io/ppforest2/main/r/reference/pp_pda.md),
[`dr_uniform`](https://andres-vidal.github.io/ppforest2/main/r/reference/dr_uniform.md)

## Examples

``` r
sr_mean_of_means()
#> $name
#> [1] "mean_of_means"
#> 
#> $display_name
#> [1] "Mean of means"
#> 
#> attr(,"class")
#> [1] "sr_strategy"
```
