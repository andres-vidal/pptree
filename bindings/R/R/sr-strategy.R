#' Mean-of-means split rule strategy.
#'
#' Creates a split rule strategy that splits at the midpoint between group
#' means. This is the default (and currently only) split rule.
#'
#' @return A \code{sr_strategy} object.
#' @seealso \code{\link{pptr}}, \code{\link{pprf}}, \code{\link{pp_pda}}, \code{\link{dr_uniform}}
#' @examples
#' sr_mean_of_means()
#'
#' @export
sr_mean_of_means <- function() {
  structure(list(name = "mean_of_means", display_name = "Mean of means"), class = "sr_strategy")
}
