#' Largest-gap binarization strategy.
#'
#' Creates a binarization strategy that reduces multiclass nodes to binary
#' by projecting group means and splitting at the largest gap. This is the
#' default binarization strategy.
#'
#' @return A \code{binarize_strategy} object.
#' @seealso \code{\link{pptr}}, \code{\link{pprf}}
#' @examples
#' binarize_largest_gap()
#'
#' @export
binarize_largest_gap <- function() {
  structure(list(name = "largest_gap", display_name = "Largest gap"), class = "binarize_strategy")
}
