#' Majority-vote leaf strategy.
#'
#' Creates a leaf strategy that assigns the majority group label as the
#' leaf prediction. When groups are tied, the smallest label wins.
#' This is the default leaf strategy.
#'
#' @return A \code{leaf_strategy} object.
#' @seealso \code{\link{pptr}}, \code{\link{pprf}}
#' @examples
#' leaf_majority_vote()
#'
#' @export
leaf_majority_vote <- function() {
  structure(list(name = "majority_vote", display_name = "Majority vote"), class = "leaf_strategy")
}
