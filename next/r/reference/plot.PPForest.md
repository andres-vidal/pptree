# Plot a PPForest model.

Visualizes a PPForest model. By default, shows variable importance
across all available metrics. Use `type` to show individual plots.

## Usage

``` r
# S3 method for class 'PPForest'
plot(x, type = "importance", tree_index = 1L, node = 1L, ...)
```

## Arguments

- x:

  A PPForest model.

- type:

  Character string specifying the plot type. `"importance"` (default)
  shows variable importance (all available metrics), `"structure"` shows
  a specific tree with embedded histograms, `"projection"` shows
  projected data at a node, `"boundaries"` shows decision boundaries of
  a specific tree.

- tree_index:

  Integer index of the tree to plot (1-based). Only used when
  `type = "structure"`, `type = "projection"`, or `type = "boundaries"`.
  Defaults to 1.

- node:

  Integer index of the node for projection plots. Defaults to 1 (root).
  Only used when `type = "projection"`.

- ...:

  Additional arguments passed to the internal plotting function.

## Value

A ggplot2 object (invisibly).

## Examples

``` r
if (FALSE) { # \dontrun{
forest <- PPForest(Type ~ ., data = iris, size = 10)
plot(forest)
plot(forest, type = "structure", tree_index = 1)
plot(forest, type = "projection", tree_index = 1)
} # }
```
