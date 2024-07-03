library(microbenchmark)
library(plyr)
library(magrittr)


#----------
# Functions
#----------

dt_fn <- function(g, n, p) {
  # g: groups; n: obs per group; p: predictor variables
  x <- matrix(rnorm(p * n * g), ncol = p, nrow = n * g)
  data.frame(type = as.factor(rep(1:g, each = n)), x)
}

bsq_fn <- function(dd, mtree, cr) {
  # dd: data, mtree: number of trees, #cr number of parallel cores.
  p <- ncol(dd[, -1])
  PPforest(
    data = dd, class = "type",
    size.tr = 1, m = mtree, size.p = sqrt(p - 1) / (p - 1),
    PPmethod = "PDA"
  )
}

bsqOld_fn <- function(dd, mtree, cr) {
  # dd: data, mtree: number of trees, #cr number of parallel cores.
  p <- ncol(dd[, -1])
  PPforest(
    data = dd, class = "type",
    size.tr = 1, m = mtree, size.p = sqrt(p - 1) / (p - 1),
    PPmethod = "PDA", strata = TRUE
  )
}


bsqHP_fn <- function(dd, mtree, cr) {
  # dd: data, mtree: number of trees, #cr number of parallel cores.
  p <- ncol(dd[, -1])
  PPForest(type ~ ., data = dd, size = mtree, lambda = .1, n_vars = round((sqrt(p - 1) / (p - 1)) * p))
}

# Ejemplo



micro_fn <- function(gs, ns, ps, mtrees, crs, version, dd.list) {
  # identify dataset
  # dd.list<-  sticky_all(dd.list)
  ids <- attributes(dd.list)$split_labels |>
    dplyr::mutate(id = 1:length(dd.list)) |>
    dplyr::filter(g == gs, n == ns, p == ps)

  # run forest
  if (version == "new") {
    md <- microbenchmark(bsq_fn(dd = dts[[ids$id]], mtree = mtrees), times = 5)
    # } else if(version == 'old') {
    #   md = microbenchmark( bsqOld_fn(dd = dts[[ids$id]], mtree = mtrees), times = 5)
  } else if (version == "HP") {
    md <- microbenchmark(bsqHP_fn(dd = dts[[ids$id]], mtree = mtrees), times = 5)
  }
  return(md)
}

# restart R session, and run only the 'functions' and packages

# datasets
prs.dt <- expand.grid(g = c(3, 6, 9), n = c(10, 100), p = c(10, 100))
dts <- mlply(prs.dt, dt_fn)

# set up scenarios, separating new and HP
prs.new <- expand.grid(
  gs = c(3, 6, 9), ns = c(10, 100),
  ps = c(10, 100), mtrees = c(50, 500), version = "new"
)
# prs.old <- expand.grid(gs = c(3, 6, 9), ns = c(10, 100),
#                        ps = c(10, 100), mtrees = c(50, 500), version = 'old' )

prs.HP <- expand.grid(
  gs = c(3, 6, 9), ns = c(10, 100),
  ps = c(10, 100), mtrees = c(50, 500), version = "HP"
)
pt <- proc.time()
# run NEW scenarios
library(PPforest)
mm.new <- mdply(prs.new, micro_fn, dd.list = dts)

# run OLD scenarios
detach("package:PPforest", unload = TRUE)
# devtools::install_github("natydasilva/PPforest_old")
# library(PPforestold)
# mm.old <- mdply(prs.old, micro_fn, dd.list=dts)

library(PPTree)
mm.HP <- mdply(prs.HP, micro_fn, dd.list = dts)

pt <- proc.time() - pt
save(mm.new, mm.HP, file = "preformance_timesWT.Rdata")

### VIZ

load("preformance_timesWT.Rdata")
library(tidyverse)
library(magrittr)

global_labeller <- labeller(
  g = class,
  ntrees = trees,
  .default = label_both
)

mm.allbig <- rbind(mm.new, mm.HP) %>%
  dplyr::mutate(seconds = time / 1e9, prop.vs = as.factor(version))

mm.allbig <- mm.allbig %>%
  rename(n = ns, p = ps, g = gs, B = mtrees)
ggplot(data = mm.allbig) +
  geom_smooth(aes(g, seconds,
    color = prop.vs,
    linetype = prop.vs
  ), method = "lm") +
  geom_jitter(aes(g, seconds, color = prop.vs, shape = prop.vs),
    alpha = .4, size = 3, height = 0
  ) +
  scale_x_continuous(breaks = seq(0, 10, 2)) +
  labs(x = "Num. groups", y = "Time (sec)") +
  scale_colour_viridis_d(
    name = "Proportion of variables",
    begin = 0.1, end = 0.6,
    option = "magma"
  ) +
  scale_shape(name = "Proportion of variables") +
  scale_linetype(name = "Proportion of variables") +
  facet_grid(B ~ n + p,
    labeller = label_both,
    scales = "free_y"
  ) +
  theme(
    legend.position = "bottom",
    axis.text = element_text(size = 6), aspect.ratio = 1
  )
