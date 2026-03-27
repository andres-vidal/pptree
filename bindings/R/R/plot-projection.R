# ======================================================================
# Projection histogram plot
# ======================================================================

#' Retrieve a tree node by breadth-first (BFS) index.
#'
#' Traverses the tree in BFS order and returns the node at position
#' \code{index} (1-based).  Used to locate the target node for projection
#' plots when the user specifies a node number.
#'
#' @param root The root node of the tree (\code{$root} from a pptr model).
#' @param index 1-based BFS index (1 = root, 2 = first child, etc.).
#' @return The tree node at the specified BFS position.
#' @note Stops with an error if the index is out of range.
#' @noRd
get_node_bfs <- function(root, index) {
  queue <- list(root)
  i <- 0L

  while (length(queue) > 0) {
    current <- queue[[1]]
    queue <- queue[-1]
    i <- i + 1L

    if (i == index) return(current)

    if (!is.null(current$lower)) queue <- c(queue, list(current$lower))
    if (!is.null(current$upper)) queue <- c(queue, list(current$upper))
  }

  stop("Node index ", index, " is out of range.", call. = FALSE)
}

#' Render a projection histogram at a specified split node.
#'
#' Projects all training observations onto the node's projector vector,
#' draws overlapping histograms coloured by group, and marks the split
#' threshold with a dashed vertical line.  This shows how well the
#' node's projection separates the groups.
#'
#' @param model A pptr model with \code{$root}, \code{$x}, \code{$y},
#'   \code{$groups}.
#' @param node 1-based BFS index of the target node (default: root).
#' @param ... Currently unused.
#' @return A ggplot2 object.
#' @note Stops with an error if the specified node is a leaf.
#' @noRd
plot_projection <- function(model, node = 1L, ...) {
  target_node <- get_node_bfs(model$root, node)

  if (!is.null(target_node$value)) {
    stop("Node ", node, " is a leaf node and has no projector.", call. = FALSE)
  }

  projector <- target_node$projector
  projected <- as.numeric(model$x %*% projector)
  group_labels <- model$groups[model$y]

  df <- data.frame(
    projected = projected,
    group     = group_labels,
    stringsAsFactors = FALSE
  )

  threshold <- target_node$threshold

  ggplot2::ggplot(df, ggplot2::aes(x = projected, fill = group)) +
    ggplot2::geom_histogram(
      bins     = 30,
      alpha    = ppforest2_alpha_proj,
      position = "identity"
    ) +
    ggplot2::geom_vline(
      xintercept = threshold,
      linetype   = "dashed",
      color      = ppforest2_col_threshold,
      linewidth  = ppforest2_lw_medium
    ) +
    ggplot2::labs(
      title = paste0("Projected Data at Node ", node),
      x     = "Projected Value",
      y     = "Count",
      fill  = "Class"
    ) +
    ppforest2_theme()
}
