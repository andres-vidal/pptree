# ======================================================================
# Mosaic layout
# ======================================================================

#' Render a composite mosaic of tree structure, importance, and boundaries.
#'
#' Arranges three plots in a 2-column grid (AB / AC design):
#' \itemize{
#'   \item Left column (full height): tree structure diagram (no legend)
#'   \item Top-right: variable importance bar chart
#'   \item Bottom-right: decision boundaries with Class legend at bottom
#' }
#'
#' Column widths adapt to the tree's x-span.  Row heights adapt to the
#' number of features.
#'
#' @param model A pptr model.
#' @param ... Passed through to \code{plot_boundaries()}.
#' @return A patchwork object (ggplot-compatible, works with ggsave).
#' @noRd
plot_mosaic <- function(model, ...) {
  check_patchwork()

  mosaic_title_size <- 11
  mosaic_margin <- ggplot2::theme(
    plot.title = ggplot2::element_text(size = mosaic_title_size),
    plot.margin = ggplot2::margin(5, 5, 5, 5)
  )

  p_structure <- plot_tree_structure(model) +
    mosaic_margin +
    ggplot2::theme(legend.position = "none")

  p_importance <- plot_importance(model) + mosaic_margin

  # Compact mode: no aspect ratio, no axis text, upper triangle blanked.
  # Pass title size so it matches the other mosaic titles.
  p_boundaries <- plot_boundaries(model, compact = mosaic_title_size, ...)
  if (inherits(p_boundaries, "gtable")) {
    p_boundaries <- patchwork::wrap_elements(full = p_boundaries)
  } else {
    p_boundaries <- p_boundaries + mosaic_margin
  }

  tree_frac <- 0.60

  n_features <- ncol(model$x)
  imp_ratio <- 1
  bnd_ratio <- if (n_features <= 2) 2 else if (n_features <= 4) 3 else 4

  p_structure + p_importance + p_boundaries +
    patchwork::plot_layout(
      design  = "AB\nAC",
      widths  = c(tree_frac, 1 - tree_frac),
      heights = c(imp_ratio, bnd_ratio)
    )
}
