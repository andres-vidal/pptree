# Group-based partition strategy.

Creates a partition strategy that routes all observations of a group to
the same child node. This is the default partition strategy.

## Usage

``` r
partition_by_group()
```

## Value

A `partition_strategy` object.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/main/r/reference/pptr.md),
[`pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/pprf.md)

## Examples

``` r
partition_by_group()
#> $name
#> [1] "by_group"
#> 
#> $display_name
#> [1] "By group"
#> 
#> attr(,"class")
#> [1] "partition_strategy"
```
