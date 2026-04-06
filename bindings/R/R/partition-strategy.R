#' Group-based partition strategy.
#'
#' Creates a partition strategy that routes all observations of a group to the
#' same child node. This is the default partition strategy.
#'
#' @return A \code{partition_strategy} object.
#' @seealso \code{\link{pptr}}, \code{\link{pprf}}
#' @examples
#' partition_by_group()
#'
#' @export
partition_by_group <- function() {
  structure(list(name = "by_group", display_name = "By group"), class = "partition_strategy")
}
