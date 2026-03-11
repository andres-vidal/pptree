# Plot a PPTree model.

Visualizes a PPTree model. By default, shows a mosaic overview with tree
structure, decision boundaries, and variable importance. Use `type` to
show individual plots.

## Usage

``` r
# S3 method for class 'PPTree'
plot(x, type = NULL, node = 1L, ...)
```

## Arguments

- x:

  A PPTree model.

- type:

  Character string specifying the plot type. `NULL` (default) shows a
  mosaic overview. Other options: `"structure"` for tree with embedded
  histograms, `"importance"` for variable importance (all available
  metrics), `"projection"` for projected data at a node, `"boundaries"`
  for decision boundaries in feature space.

- node:

  Integer index of the node for projection plots (1-based, breadth-first
  order). Defaults to 1 (root node). Only used when
  `type = "projection"`.

- ...:

  Additional arguments passed to the internal plotting function.

## Value

A ggplot2 object (invisibly), or `NULL` for the mosaic layout.

## Examples

``` r
if (FALSE) { # \dontrun{
model <- PPTree(Type ~ ., data = iris)
plot(model)                         # mosaic overview
plot(model, type = "structure")     # tree structure only
plot(model, type = "importance")    # variable importance
plot(model, type = "projection")   # projection histogram
plot(model, type = "boundaries")   # decision boundaries
} # }
```
