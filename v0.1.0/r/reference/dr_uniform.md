# Uniform random variable selection strategy.

Creates a dimensionality reduction strategy that randomly selects a
subset of variables at each split. Used with
[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pprf.md)
for random forests.

## Usage

``` r
dr_uniform(n_vars = NULL, p_vars = NULL)
```

## Arguments

- n_vars:

  The number of variables to consider at each split (integer). Cannot be
  used together with `p_vars`.

- p_vars:

  The proportion of variables to consider at each split (number between
  0 and 1, exclusive). Resolved to an integer count when the number of
  features is known. Cannot be used together with `n_vars`.

## Value

A `dr_strategy` object.

## See also

[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pprf.md),
[`dr_noop`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/dr_noop.md),
[`pp_pda`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pp_pda.md),
[`sr_mean_of_means`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/sr_mean_of_means.md)

## Examples

``` r
# Select 2 variables at each split
dr_uniform(n_vars = 2)
#> $name
#> [1] "uniform"
#> 
#> $display_name
#> [1] "Uniform random"
#> 
#> $n_vars
#> [1] 2
#> 
#> $p_vars
#> NULL
#> 
#> attr(,"class")
#> [1] "dr_strategy"

# Select half the variables at each split
dr_uniform(p_vars = 0.5)
#> $name
#> [1] "uniform"
#> 
#> $display_name
#> [1] "Uniform random"
#> 
#> $n_vars
#> NULL
#> 
#> $p_vars
#> [1] 0.5
#> 
#> attr(,"class")
#> [1] "dr_strategy"

# Use with pprf
pprf(Type ~ ., data = iris, dr = dr_uniform(n_vars = 2))
#> 
#> Random Forest of Project-Pursuit Oblique Decision Tree
#> -------------------------------------
#> Tree 1:
#> If ([ 0.01 0 0 0.1 ] * x) < 0.1471151:
#>   Predict: setosa 
#> Else:
#>  If ([ 0.01 0 0 0.24 ] * x) < 0.4446007:
#>    Predict: versicolor 
#>  Else:
#>    Predict: virginica 
#> 
#> Tree 2:
#> If ([ -0.07 0.11 0 0 ] * x) < -0.07210007:
#>  If ([ 0 0.04 -0.12 0 ] * x) < -0.4739328:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
#> 
```
