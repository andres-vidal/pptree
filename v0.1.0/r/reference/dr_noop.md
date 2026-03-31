# No-op dimensionality reduction strategy.

Creates a dimensionality reduction strategy that uses all variables at
each split. This is the default for single trees
([`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pptr.md)).

## Usage

``` r
dr_noop()
```

## Value

A `dr_strategy` object.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pptr.md),
[`dr_uniform`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/dr_uniform.md),
[`pp_pda`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pp_pda.md),
[`sr_mean_of_means`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/sr_mean_of_means.md)

## Examples

``` r
dr_noop()
#> $name
#> [1] "noop"
#> 
#> $display_name
#> [1] "All variables"
#> 
#> attr(,"class")
#> [1] "dr_strategy"
```
