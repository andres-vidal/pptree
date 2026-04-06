#' Mean-of-means split cutpoint strategy.
#'
#' Creates a split cutpoint strategy that splits at the midpoint between group
#' means. This is the default (and currently only) split cutpoint.
#'
#' @return A \code{cutpoint_strategy} object.
#' @seealso \code{\link{pptr}}, \code{\link{pprf}}, \code{\link{pp_pda}}, \code{\link{vars_uniform}}
#' @examples
#' cutpoint_mean_of_means()
#'
#' @export
cutpoint_mean_of_means <- function() {
  structure(list(name = "mean_of_means", display_name = "Mean of means"), class = "cutpoint_strategy")
}
