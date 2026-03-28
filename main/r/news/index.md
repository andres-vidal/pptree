# Changelog

## ppforest2 0.1.0

### New features

- Core: Projection-pursuit oblique decision trees and random forests for
  classification, using LDA/PDA optimization.
- Core: Random uniform variable selection per split for forest
  diversity.
- Core: Three variable importance measures: permuted (VI1), projections
  (VI2), and weighted projections (VI3).
- Core: Out-of-bag error and confusion matrix for forests, with
  bootstrap sample indices persisted for recomputation.
- Core: Degenerate split detection when projection-pursuit cannot find a
  useful projection.
- Core: OpenMP multi-threaded forest training.
- Core: JSON serialization for trained models.
- Core: Cross-platform reproducibility: identical results for the same
  seed on Linux, macOS, and Windows.
- R:
  [`pptr()`](https://andres-vidal.github.io/ppforest2/main/r/reference/pptr.md)
  and
  [`pprf()`](https://andres-vidal.github.io/ppforest2/main/r/reference/pprf.md)
  with formula and matrix interfaces.
- R: [`predict()`](https://rdrr.io/r/stats/predict.html) with group
  labels (`type = "class"`) or vote proportions (`type = "prob"`).
- R: Training and OOB confusion matrices displayed by
  [`summary()`](https://rdrr.io/r/base/summary.html).
- R: Degenerate split warnings when projection-pursuit cannot separate
  groups.
- R:
  [`save_json()`](https://andres-vidal.github.io/ppforest2/main/r/reference/save_json.md)
  and
  [`load_json()`](https://andres-vidal.github.io/ppforest2/main/r/reference/load_json.md)
  for model persistence.
- R: tidymodels/parsnip integration:
  [`pp_tree()`](https://andres-vidal.github.io/ppforest2/main/r/reference/pp_tree.md)
  and
  [`pp_rand_forest()`](https://andres-vidal.github.io/ppforest2/main/r/reference/pp_rand_forest.md)
  model specifications.
- R: ggplot2 visualizations: tree diagrams, variable importance plots,
  projection histograms, and decision boundary plots.
- R: Bundled datasets: crab, crabs, fishcatch, glass, image, iris,
  leukemia, lymphoma, NCI60, olive, parkinson, and wine.
- CLI: `train`: fit a tree or forest from CSV, save as JSON.
- CLI: `predict`: classify new data with a saved model.
- CLI: `evaluate`: train/test evaluation with smart convergence.
- CLI: `summarize`: display model configuration, data summary, and
  metrics from a saved model JSON. Supports `--data` to recompute
  metrics from training data.
- CLI: `benchmark`: multi-scenario performance benchmarks with baseline
  comparison.
