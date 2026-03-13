# ======================================================================
# Mosaic layout
# ======================================================================

#' Render a composite mosaic of tree structure, importance, and boundaries.
#'
#' Arranges three plots in a 2-column grid layout using the grid package:
#' \itemize{
#'   \item Left column (full height): tree structure diagram
#'   \item Top-right: variable importance bar chart
#'   \item Bottom-right: decision boundaries
#' }
#'
#' Layout proportions adapt to the data:
#' \itemize{
#'   \item Column widths are computed from the tree's x-span (wider trees
#'     get a wider left column, clamped to 35--60\% of total width)
#'   \item Row heights adapt to the number of features: fewer features give
#'     more vertical space to importance, more features give more space to
#'     the pairwise boundary grid
#' }
#'
#' @param model A pptr model.
#' @param ... Passed through to \code{plot_boundaries()}.
#' @return NULL (invisibly). The plot is drawn as a side effect via grid.
#' @noRd
plot_mosaic <- function(model, ...) {
  p_structure <- plot_tree_structure(model) +
    ggplot2::theme(plot.margin = ggplot2::margin(5, 10, 5, 5))
  p_importance <- plot_importance(model) +
    ggplot2::theme(plot.margin = ggplot2::margin(5, 5, 5, 10))
  p_boundaries <- plot_boundaries(model, ...)
  if (inherits(p_boundaries, "ggplot")) {
    p_boundaries <- p_boundaries +
      ggplot2::theme(plot.margin = ggplot2::margin(5, 5, 5, 10))
  }

  # Compute adaptive layout proportions from the data
  layout <- ppforest2_tree_layout(model)
  node_df <- layout$nodes
  tree_x_span <- diff(range(node_df$x)) + ppforest2_node_w + 0.4
  tree_frac <- max(0.35, min(0.60, tree_x_span / (tree_x_span + 3.0)))

  n_features <- ncol(model$x)
  imp_ratio <- if (n_features <= 2) 2 else if (n_features <= 4) 1 else 1
  bnd_ratio <- if (n_features <= 2) 1.5 else if (n_features <= 4) 2 else 3

  grid::grid.newpage()

  # Left column: structure (full height)
  # Right column: importance (top) + boundaries (bottom)
  grid::pushViewport(grid::viewport(
    layout = grid::grid.layout(2, 2,
      widths  = grid::unit(c(tree_frac, 1 - tree_frac), "null"),
      heights = grid::unit(c(imp_ratio, bnd_ratio), "null"))
  ))

  grid::pushViewport(grid::viewport(layout.pos.row = 1:2, layout.pos.col = 1))
  print(p_structure, newpage = FALSE)
  grid::popViewport()

  grid::pushViewport(grid::viewport(layout.pos.row = 1, layout.pos.col = 2))
  print(p_importance, newpage = FALSE)
  grid::popViewport()

  grid::pushViewport(grid::viewport(layout.pos.row = 2, layout.pos.col = 2))
  if (inherits(p_boundaries, "gtable")) {
    grid::grid.draw(p_boundaries)
  } else {
    print(p_boundaries, newpage = FALSE)
  }
  grid::popViewport()

  grid::popViewport()
  invisible(NULL)
}
