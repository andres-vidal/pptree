library(purrr)

bgss <- function(x, y) {
  global_mean <- attr(scale(x, scale = FALSE), "scaled:center")

  group_sum_of_squares <- function(group) {
    group_data <- x[y == group, ]
    group_mean <- attr(scale(group_data, scale = FALSE), "scaled:center")
    diff <- group_mean - global_mean
    nrow(group_data) * outer(diff, diff)
  }

  unique(y) %>%
    purrr::map(group_sum_of_squares) %>%
    purrr::reduce(`+`)
}


wgss <- function(x, y) {
  group_sum_of_squares <- function(group) {
    group_data <- x[y == group, ]
    (nrow(group_data) - 1) * cov(group_data)
  }

  unique(y) %>%
    purrr::map(group_sum_of_squares) %>%
    purrr::reduce(`+`)
}
