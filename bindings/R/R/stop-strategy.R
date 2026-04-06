#' Pure-node stopping rule.
#'
#' Creates a stopping rule that stops splitting when a node contains only one
#' group. This is the default stopping rule.
#'
#' @return A \code{stop_strategy} object.
#' @seealso \code{\link{pptr}}, \code{\link{pprf}}
#' @examples
#' stop_pure_node()
#'
#' @export
stop_pure_node <- function() {
  structure(list(name = "pure_node", display_name = "Pure node"), class = "stop_strategy")
}
