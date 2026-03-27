# ======================================================================
# Decision boundary plots
# ======================================================================

#' Dispatch to the appropriate boundary plot based on feature count.
#'
#' Selects the visualization strategy based on the number of features (p):
#' \describe{
#'   \item{p = 1}{1D number-line plot.  Background rectangles show decision
#'     regions; solid vertical lines mark boundaries; jittered points show
#'     the data.}
#'   \item{p = 2}{Single 2D scatterplot.  Convex region polygons (from C++
#'     Sutherland-Hodgman clipping) fill the background; boundary line
#'     segments are drawn on top; data points are overlaid.  Uses
#'     \code{coord_fixed(ratio = 1)} for undistorted angles.}
#'   \item{p >= 3}{Pairwise scatterplot matrix (lower triangle).  Each panel
#'     shows a 2D projection with non-displayed variables held at their
#'     medians.  \code{facet_grid} with free scales and zero expansion
#'     ensures full region coverage per panel.}
#' }
#'
#' @param model A pptr model with \code{$root}, \code{$x}, \code{$y},
#'   \code{$groups}.
#' @param ... Passed through to the specific boundary plot function.
#' @return A ggplot2 object.
#' @noRd
plot_boundaries <- function(model, ...) {
  p <- ncol(model$x)

  if (p == 1L) {
    plot_boundaries_1d(model)
  } else if (p == 2L) {
    plot_boundaries_2d(model, ...)
  } else {
    plot_boundaries_pairwise(model, ...)
  }
}

#' Convert C++ region polygon data to a ggplot2-ready data frame.
#'
#' Transforms the list-of-lists output from \code{ppforest2_decision_regions()}
#' (C++ RegionVisitor) into a flat data frame suitable for
#' \code{ggplot2::geom_polygon()}.  Each polygon becomes a group of rows
#' sharing a unique region_id, with vertices in the order produced by the
#' Sutherland-Hodgman algorithm (consistent winding for correct polygon
#' rendering).
#'
#' Group indices from C++ are 1-based (matching R's convention for the
#' response vector) and are mapped to human-readable labels via the
#' model's group vector.
#'
#' @param regions List of region lists, each with \code{$x}, \code{$y}
#'   (numeric vectors of vertex coordinates) and \code{$group} (integer
#'   group index, 1-based from C++).
#' @param group_labels Character vector of group names (model$groups).
#' @return Data frame with columns x, y, region_id, region_group, or
#'   NULL if no valid polygons exist.  Degenerate regions with fewer
#'   than 3 vertices are silently dropped.
#' @noRd
build_region_df <- function(regions, group_labels) {
  if (length(regions) == 0) return(NULL)

  dfs <- list()
  for (i in seq_along(regions)) {
    r <- regions[[i]]
    if (length(r$x) >= 3) {
      dfs[[length(dfs) + 1L]] <- data.frame(
        x = r$x, y = r$y,
        region_id = i,
        region_group = group_labels[r$group],
        stringsAsFactors = FALSE
      )
    }
  }

  if (length(dfs) == 0) return(NULL)
  do.call(rbind, dfs)
}

#' Render decision boundaries for a 1-feature model.
#'
#' Draws a number-line plot with three layers (back to front):
#' \enumerate{
#'   \item Coloured background rectangles for each decision region
#'     (predicted group determined by running \code{ppforest2_predict_tree} on each
#'     region's midpoint)
#'   \item Jittered data points coloured by true group (y is meaningless,
#'     jittered around 0 for visual separation)
#'   \item Solid vertical lines at each split's decision boundary
#' }
#'
#' Boundary positions are computed as \code{threshold / projector[1]} for
#' each internal node.  For a 1D projector \code{a = [a1]}, the split
#' condition \code{a^T x < t} becomes \code{a1 * x < t}, i.e.
#' \code{x < t/a1}.  The sorted boundary positions partition the x-axis
#' into contiguous regions, each assigned the group predicted by the tree
#' for that region's midpoint.
#'
#' The y-axis is suppressed (no title, no ticks, no labels) since the
#' vertical dimension has no meaning — it exists only to spread out the
#' jittered points for readability.
#'
#' @param model A pptr model with 1 feature column in \code{$x}.
#' @return A ggplot2 object (no y-axis).
#' @noRd
plot_boundaries_1d <- function(model) {
  vnames <- get_variable_names(model)
  all_group_labels <- model$groups
  group_colors <- get_group_colors(all_group_labels)
  obs_group_labels <- all_group_labels[model$y]

  nodes <- ppforest2_tree_node_data(model, model$x, model$y)

  boundary_x <- vapply(
    Filter(function(nd) !nd$is_leaf, nodes),
    function(nd) nd$threshold / nd$projector[1],
    numeric(1)
  )
  boundary_x <- sort(boundary_x)

  x_range <- range(model$x[, 1])
  x_pad <- diff(x_range) * 0.05
  x_range <- x_range + c(-x_pad, x_pad)

  # Build region rectangles from sorted boundaries
  # Predict the group for the midpoint of each region
  edges <- c(x_range[1], boundary_x, x_range[2])
  region_rects <- list()
  for (r in seq_len(length(edges) - 1)) {
    mid <- (edges[r] + edges[r + 1]) / 2
    mid_vec <- matrix(mid, nrow = 1)
    pred_group <- all_group_labels[ppforest2_predict_tree(model, mid_vec)]
    region_rects[[r]] <- data.frame(
      xmin = edges[r], xmax = edges[r + 1],
      region_group = pred_group,
      stringsAsFactors = FALSE
    )
  }
  region_df <- do.call(rbind, region_rects)

  obs_data <- data.frame(
    x     = model$x[, 1],
    group = obs_group_labels,
    stringsAsFactors = FALSE
  )

  # Make the plot square: y-range equals x-range width, centred at 0
  x_span <- diff(x_range)
  y_range <- c(-x_span / 2, x_span / 2)

  ggplot2::ggplot() +
    ggplot2::geom_rect(
      data = region_df,
      ggplot2::aes(xmin = xmin, xmax = xmax, ymin = y_range[1], ymax = y_range[2],
                   fill = region_group),
      alpha = ppforest2_alpha_region
    ) +
    ggplot2::geom_jitter(
      data   = obs_data,
      ggplot2::aes(x = x, y = 0, color = group),
      height = 0.3, size = ppforest2_pt_medium
    ) +
    ggplot2::geom_vline(
      xintercept = boundary_x,
      linetype   = "solid",
      color      = ppforest2_col_boundary,
      linewidth  = ppforest2_lw_medium
    ) +
    ggplot2::scale_fill_manual(values = group_colors, name = "Class") +
    ggplot2::scale_color_manual(values = group_colors, name = "Class") +
    ggplot2::scale_x_continuous(expand = ggplot2::expansion(0)) +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(0)) +
    ggplot2::labs(
      title = "Decision Boundaries",
      x     = vnames[1]
    ) +
    ggplot2::coord_cartesian(xlim = x_range, ylim = y_range) +
    ppforest2_theme() +
    ggplot2::theme(
      aspect.ratio = 1,
      axis.title.y = ggplot2::element_blank(),
      axis.text.y  = ggplot2::element_blank(),
      axis.ticks.y = ggplot2::element_blank()
    )
}

#' Render decision boundaries for a 2-feature model.
#'
#' Draws a single scatterplot with:
#' \itemize{
#'   \item Convex region polygons (Sutherland-Hodgman clipped) coloured by
#'     predicted group, computed by \code{ppforest2_decision_regions()} in C++
#'   \item Boundary line segments (parametric-clipped) from
#'     \code{ppforest2_boundary_segments()} in C++
#'   \item Data points coloured by true group
#' }
#'
#' The plot uses \code{coord_fixed(ratio = 1)} with equalized ranges so
#' the decision boundaries appear at their true angles.  Region polygons
#' are computed on an extended bounding box (0.5x padding) so they reach
#' the plot edges.
#'
#' @param model A pptr model with 2 feature columns in \code{$x}.
#' @param ... Passed to \code{geom_segment} (e.g. additional aesthetics).
#' @return A ggplot2 object.
#' @noRd
plot_boundaries_2d <- function(model, ...) {
  vnames <- get_variable_names(model)
  group_labels <- model$groups
  group_colors <- get_group_colors(group_labels)

  obs_group_labels <- group_labels[model$y]

  x_range <- range(model$x[, 1])
  y_range <- range(model$x[, 2])
  x_pad <- diff(x_range) * 0.05
  y_pad <- diff(y_range) * 0.05
  x_range <- x_range + c(-x_pad, x_pad)
  y_range <- y_range + c(-y_pad, y_pad)

  # Equalize ranges so the plot is square with coord_fixed
  eq <- equalize_ranges(x_range, y_range)
  x_range <- eq$x
  y_range <- eq$y

  # Use a larger bounding box for regions so they extend to the plot edges
  x_ext <- diff(x_range) * 0.5
  y_ext <- diff(y_range) * 0.5
  region_bbox <- c(x_range[1] - x_ext, x_range[2] + x_ext,
                   y_range[1] - y_ext, y_range[2] + y_ext)

  # Decision regions (computed over extended bbox, clipped visually by coord)
  regions <- ppforest2_decision_regions(
    model, c(0L, 1L), numeric(0),
    region_bbox[1], region_bbox[2], region_bbox[3], region_bbox[4]
  )
  region_df <- build_region_df(regions, group_labels)

  # Boundary segments
  segments <- ppforest2_boundary_segments(
    model, c(0L, 1L), numeric(0),
    region_bbox[1], region_bbox[2], region_bbox[3], region_bbox[4]
  )

  obs_data <- data.frame(
    x     = model$x[, 1],
    y     = model$x[, 2],
    group = obs_group_labels,
    stringsAsFactors = FALSE
  )

  p <- ggplot2::ggplot()

  # Regions
  if (!is.null(region_df)) {
    p <- p + ggplot2::geom_polygon(
      data = region_df,
      ggplot2::aes(x = x, y = y, group = region_id, fill = region_group),
      alpha = ppforest2_alpha_region
    )
  }

  # Boundary lines
  if (nrow(segments) > 0) {
    p <- p + ggplot2::geom_segment(
      data = segments,
      ggplot2::aes(x = x_start, y = y_start, xend = x_end, yend = y_end),
      linewidth = ppforest2_lw_medium, color = ppforest2_col_boundary, ...
    )
  }

  # Data points
  p <- p + ggplot2::geom_point(
    data = obs_data,
    ggplot2::aes(x = x, y = y, color = group),
    size = ppforest2_pt_medium
  )

  p +
    ggplot2::scale_fill_manual(values = group_colors, name = "Class") +
    ggplot2::scale_color_manual(values = group_colors, name = "Class") +
    ggplot2::labs(
      title = "Decision Boundaries",
      x     = vnames[1],
      y     = vnames[2]
    ) +
    ggplot2::coord_fixed(ratio = 1, xlim = x_range, ylim = y_range) +
    ppforest2_theme()
}

#' Render pairwise decision boundaries for a model with 3+ features.
#'
#' Creates a lower-triangular scatterplot matrix using \code{facet_grid()}.
#' For each pair of features (i, j), all other features are held at
#' their median values and the tree's decision regions and boundary
#' lines are projected onto the (i, j) plane.
#'
#' Layout uses \code{facet_grid(y_var ~ x_var, switch = "both")} so that
#' column strips appear at the bottom (serving as x-axis labels) and
#' row strips appear on the left (serving as y-axis labels), similar
#' to a \code{pairs()} plot.  All panels use free scales and
#' \code{aspect.ratio = 1}.
#'
#' @section Full region coverage:
#' The region bounding box is set to exactly the padded data range
#' (data range +/- 5\%).  Scale expansion is disabled via
#' \code{expansion(0)}, so the visible coordinate range equals the
#' polygon bounding box.  Since the Sutherland-Hodgman algorithm
#' partitions the entire bbox across all leaves, the region polygons
#' tile the panel completely with no whitespace gaps.
#'
#' @section Axis labels:
#' The overall x/y axis titles are suppressed.  The strip labels
#' (placed at the axis positions via \code{switch = "both"}) serve as
#' the axis labels, showing the variable name for each row and column.
#'
#' Remove empty upper-triangle panels from a pairwise facet_grid plot.
#'
#' Converts a ggplot with \code{facet_grid(y_var ~ x_var)} to a gtable
#' and replaces panel grobs in the upper triangle (where the column
#' index exceeds the row index) with \code{nullGrob()}.  This prevents
#' empty panels from showing grid lines and background.
#'
#' @param p A ggplot object produced by the pairwise boundary builder.
#' @return A gtable with upper-triangle panel grobs blanked out.
#' @noRd
strip_upper_triangle <- function(p) {
  g <- ggplot2::ggplotGrob(p)

  panels <- g$layout[grepl("^panel", g$layout$name), ]
  panel_rows <- sort(unique(panels$t))
  panel_cols <- sort(unique(panels$l))

  for (i in seq_len(nrow(panels))) {
    facet_row <- match(panels$t[i], panel_rows)
    facet_col <- match(panels$l[i], panel_cols)

    if (facet_col > facet_row) {
      g$grobs[[which(g$layout$name == panels$name[i])]] <- grid::nullGrob()
    }
  }

  g
}

#' @param model A pptr model with 3+ feature columns in \code{$x}.
#' @param ... Passed to \code{geom_segment} (e.g. additional aesthetics).
#' @return A gtable with empty upper-triangle panels removed.
#' @noRd
plot_boundaries_pairwise <- function(model, ...) {
  vnames <- get_variable_names(model)
  group_labels <- model$groups
  group_colors <- get_group_colors(group_labels)
  n_vars <- ncol(model$x)

  obs_group_labels <- group_labels[model$y]
  medians <- apply(model$x, 2, stats::median)

  # Generate all pairwise combinations (lower triangle of scatterplot matrix)
  pairs <- utils::combn(n_vars, 2, simplify = FALSE)

  # Factor levels for grid layout: x_var = vars 1..(p-1), y_var = vars 2..p
  x_var_levels <- vnames[seq_len(n_vars - 1L)]
  y_var_levels <- vnames[seq(2L, n_vars)]

  all_segments <- list()
  all_regions <- list()
  all_obs <- list()

  for (k in seq_along(pairs)) {
    i <- pairs[[k]][1]
    j <- pairs[[k]][2]
    other <- setdiff(seq_len(n_vars), c(i, j))

    x_range <- range(model$x[, i])
    y_range <- range(model$x[, j])
    x_pad <- diff(x_range) * 0.05
    y_pad <- diff(y_range) * 0.05
    x_range <- x_range + c(-x_pad, x_pad)
    y_range <- y_range + c(-y_pad, y_pad)

    # Region bbox matches the padded data range exactly.  Combined with
    # scale expansion(0) below, this ensures region polygons tile each
    # panel completely: the polygon vertices define the scale limits, and
    # zero expansion prevents ggplot2 from adding extra whitespace beyond
    # the polygon edges.
    region_bbox <- c(x_range[1], x_range[2], y_range[1], y_range[2])

    # Decision regions (extended bbox, clipped visually by facet coord)
    regs <- ppforest2_decision_regions(
      model, c(i - 1L, j - 1L), medians[other],
      region_bbox[1], region_bbox[2], region_bbox[3], region_bbox[4]
    )
    reg_df <- build_region_df(regs, group_labels)
    if (!is.null(reg_df)) {
      # Offset region_id to be unique across panels
      offset <- if (length(all_regions) > 0) {
        max(do.call(rbind, all_regions)$region_id)
      } else {
        0
      }
      reg_df$region_id <- reg_df$region_id + offset
      reg_df$x_var <- factor(vnames[i], levels = x_var_levels)
      reg_df$y_var <- factor(vnames[j], levels = y_var_levels)
      all_regions[[k]] <- reg_df
    }

    # Boundary segments (also extended)
    segs <- ppforest2_boundary_segments(
      model, c(i - 1L, j - 1L), medians[other],
      region_bbox[1], region_bbox[2], region_bbox[3], region_bbox[4]
    )

    if (nrow(segs) > 0) {
      segs$x_var <- factor(vnames[i], levels = x_var_levels)
      segs$y_var <- factor(vnames[j], levels = y_var_levels)
      all_segments[[k]] <- segs
    }

    all_obs[[k]] <- data.frame(
      x     = model$x[, i],
      y     = model$x[, j],
      group = obs_group_labels,
      x_var = factor(vnames[i], levels = x_var_levels),
      y_var = factor(vnames[j], levels = y_var_levels),
      stringsAsFactors = FALSE
    )
  }

  obs_data <- do.call(rbind, all_obs)
  seg_data <- do.call(rbind, Filter(Negate(is.null), all_segments))
  reg_data <- do.call(rbind, Filter(Negate(is.null), all_regions))

  p <- ggplot2::ggplot()

  # Regions
  if (!is.null(reg_data) && nrow(reg_data) > 0) {
    p <- p + ggplot2::geom_polygon(
      data = reg_data,
      ggplot2::aes(x = x, y = y, group = region_id, fill = region_group),
      alpha = ppforest2_alpha_region
    )
  }

  # Boundary lines
  if (!is.null(seg_data) && nrow(seg_data) > 0) {
    p <- p + ggplot2::geom_segment(
      data = seg_data,
      ggplot2::aes(x = x_start, y = y_start, xend = x_end, yend = y_end),
      linewidth = ppforest2_lw_medium, color = ppforest2_col_boundary, ...
    )
  }

  # Data points
  p <- p + ggplot2::geom_point(
    data = obs_data,
    ggplot2::aes(x = x, y = y, color = group),
    size = ppforest2_pt_small
  )

  p <- p +
    ggplot2::facet_grid(y_var ~ x_var, scales = "free", switch = "both") +
    ggplot2::scale_fill_manual(values = group_colors, name = "Class") +
    ggplot2::scale_color_manual(values = group_colors, name = "Class") +
    ggplot2::scale_x_continuous(expand = ggplot2::expansion(0)) +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(0)) +
    ggplot2::labs(title = "Decision Boundaries") +
    ppforest2_theme() +
    ggplot2::theme(
      aspect.ratio     = 1,
      strip.placement  = "outside",
      axis.title       = ggplot2::element_blank()
    )

  strip_upper_triangle(p)
}
