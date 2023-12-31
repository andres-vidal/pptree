source("stats.test.R")

lda_optimum_projector <- function(x, y) {
  W <- wgss(x, y)
  B <- bgss(x, y)

  eigen_res <- eigen(solve(W + B) %*% B)
  eigen_val <- eigen_res$values
  eigen_vec <- eigen_res$vectors

  print(eigen_val)
  print(round(t(eigen_vec), 5))

  return(eigen_vec[, which.max(abs(eigen_val))])
}

lda_index <- function(x, y, A) {
  W <- wgss(x, y)
  B <- bgss(x, y)


  denominator <- det(t(A) %*% (W + B) %*% A)

  if (denominator == 0) {
    return(0)
  }

  return(1 - det(t(A) %*% W %*% A) / denominator)
}
