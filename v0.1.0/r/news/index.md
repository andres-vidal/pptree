# Changelog

## ppforest2 0.1.0

### New features

- Core: Projection-pursuit oblique decision trees and random forests for
  classification, using LDA/PDA optimization.
- Core: Random uniform variable selection per split for forest
  diversity.
- Core: Three variable importance measures: permuted (VI1), projections
  (VI2), and weighted projections (VI3).
- Core: Out-of-bag error estimation for forests.
- Core: OpenMP multi-threaded forest training.
- Core: JSON serialization for trained models.
- Core: Cross-platform reproducibility: identical results for the same
  seed on Linux, macOS, and Windows.
- R:
  [`pptr()`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pptr.md)
  and
  [`pprf()`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pprf.md)
  with formula and matrix interfaces.
- R: [`predict()`](https://rdrr.io/r/stats/predict.html) with class
  labels (`type = "class"`) or vote proportions (`type = "prob"`).
- R: `save_json()` and `load_json()` for model persistence.
- R: tidymodels/parsnip integration:
  [`pp_tree()`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pp_tree.md)
  and
  [`pp_rand_forest()`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pp_rand_forest.md)
  model specifications.
- R: ggplot2 visualizations: tree diagrams, variable importance plots,
  projection histograms, and decision boundary plots.
- R: Bundled datasets: crab, crabs, fishcatch, glass, image, iris,
  leukemia, lymphoma, NCI60, olive, parkinson, and wine.
- CLI: `train`: fit a tree or forest from CSV, save as JSON.
- CLI: `predict`: classify new data with a saved model.
- CLI: `evaluate`: train/test evaluation with smart convergence.
- CLI: `benchmark`: multi-scenario performance benchmarks with baseline
  comparison.
