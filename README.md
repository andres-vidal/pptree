# pptree

[![C++ Tests](https://github.com/andres-vidal/pptree/actions/workflows/run-test.yml/badge.svg)](https://github.com/andres-vidal/pptree/actions/workflows/run-test.yml)
[![R Package Check](https://github.com/andres-vidal/pptree/actions/workflows/run-r-check.yml/badge.svg)](https://github.com/andres-vidal/pptree/actions/workflows/run-r-check.yml)

> **Work in progress** — this repository contains ongoing research and development work. Interfaces and behavior are expected to evolve as the project matures.

**pptree** is a fast, memory-efficient implementation of
[Projection Pursuit Random Forests](https://www.tandfonline.com/doi/full/10.1080/10618600.2020.1870480),
built on
[Projection Pursuit (oblique) Decision Trees](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-7/issue-none/PPtree-Projection-pursuit-classification-tree/10.1214/13-EJS810.full).
By learning linear projections at each split, the model captures complex structure that axis-aligned trees often miss, without sacrificing interpretability or scalability.

The project provides a high-performance C++ core with interfaces for **R** and a command-line interface (CLI), with **Python** bindings planned. In the R ecosystem, it is intended as a modern successor to
[`PPforest`](https://cran.r-project.org/web/packages/PPforest/index.html),
offering the same statistical foundations with significantly improved computational performance.

Developed as a Bachelor's thesis project in Statistics at **Universidad de la República (Uruguay)**.

**Key capabilities:** oblique splits via projection pursuit, multi-threaded forest training (OpenMP), cross-platform reproducibility with golden tests, multiple variable importance measures (projection-based, weighted, permutation), LDA/PDA optimization, OOB error estimation, and [parsnip](https://parsnip.tidymodels.org/) / tidymodels integration.

**Documentation:** [andres-vidal.github.io/pptree](https://andres-vidal.github.io/pptree/) — [C++ API Reference](https://andres-vidal.github.io/pptree/main/cpp/) (Doxygen) · [R Package Reference](https://andres-vidal.github.io/pptree/main/r/) (pkgdown)

## Quick Start

### CLI

Build and use the `pptree` command-line tool directly:

```bash
# Compile the project into the .build folder, where the `pptree` executable is.
make build

# Train a forest on a CSV dataset and save the model
pptree train --data data.csv --trees 100 --lambda 0.5 --save model.json

# Predict on new data using a saved model
pptree predict --model model.json --data test.csv

# Evaluate with smart convergence (default)
pptree evaluate --data data.csv --trees 100 --train-ratio 0.7

# Evaluate with fixed iterations (disables convergence)
pptree evaluate --data data.csv --trees 100 -i 10 --train-ratio 0.7

# Evaluate on simulated data (1000 rows, 10 features, 3 classes)
pptree evaluate --simulate 1000x10x3 --trees 50

# Run performance benchmarks across scenarios
pptree benchmark -s bench/default-scenarios.json
```

### R

Install the R package (CRAN submission is planned once the package stabilizes):

```r
# install.packages("devtools")
devtools::install_github("https://github.com/andres-vidal/pptree", ref = "main-r")
```

```r
library(PPTree)

# Single projection pursuit tree
model <- PPTree(Type ~ ., data = iris)
predict(model, iris)

# Random forest with 50 trees
forest <- PPForest(Type ~ ., data = iris, size = 50)
predict(forest, iris)
predict(forest, iris, type = "prob")   # vote proportions
summary(forest)                        # variable importance & model info
```

Works with [parsnip](https://parsnip.tidymodels.org/) / tidymodels:

```r
library(parsnip)

spec <- pp_forest(trees = 50, mtry = 2) %>% set_engine("PPTree")
fit <- spec %>% fit(Type ~ ., data = iris)
predict(fit, iris, type = "prob")
```

## Architecture

The project is organized into a shared C++ core and language-specific bindings:

- **C++ core** (`core/`) — All models, training algorithms, statistics, serialization, and CLI live here. This is the single source of truth for the implementation. External dependencies (Eigen, nlohmann/json, pcg, GoogleTest, Google Benchmark, CLI11, fmt, csv-parser) are declared in `core/Dependencies.cmake` and fetched automatically via CMake `FetchContent`.

- **R package** (`bindings/R/PPTree/`) — Thin Rcpp layer that exposes the C++ core to R. Type conversions between R and C++ types are defined in `inst/include/PPTree.h`. Roxygen documentation and parsnip integration are R-only.

- **Python bindings** — Planned.

## Prerequisites

|              | Linux            | macOS           | Windows                              |
|--------------|------------------|-----------------|--------------------------------------|
| **C++ core** | `cmake` >= 3.14, `make`, `gcc` | `cmake` >= 3.14, `make`, `clang` | `cmake` >= 3.14, `make`, MinGW `gcc` |
| **R package**| `R` >= 3.5       | `R` >= 3.5      | `R` >= 3.5, `Rtools`                |
| **R docs**   | TeX distribution with `pdflatex` | TeX distribution with `pdflatex` | TeX distribution with `pdflatex` |

For the R package, the C++ compiler must match the one R was built with (`gcc` on Linux/Windows, `clang` on macOS).

## Building and Testing

```bash
make build              # Release build (C++ core + tests)
make test               # Build and run C++ tests
make build-debug        # Debug build with AddressSanitizer
make test-debug         # Run debug tests
```

## Benchmarking

Performance benchmarks run configurable scenarios on simulated data, measuring execution time and peak RSS. Each scenario runs as a separate process for accurate per-scenario memory measurement.

### Running Benchmarks

```bash
make benchmark                     # Run default scenarios, print table
make benchmark-save                # Run and save results to bench/results.json + bench/results.csv
make benchmark-compare             # Run and compare against saved results
make benchmark-vs REF=main         # Compare current branch against another ref (branch/tag/commit)
```

### CLI Usage

```bash
# Run scenarios from a JSON file
pptree benchmark -s bench/default-scenarios.json

# Save results as JSON and CSV
pptree benchmark -s bench/default-scenarios.json -o results.json --csv results.csv

# Compare against a baseline
pptree benchmark -s bench/default-scenarios.json -b baseline.json

# Override iteration count (forces fixed mode)
pptree benchmark -s bench/default-scenarios.json -i 5
```

### Smart Convergence

The `evaluate` subcommand uses smart convergence by default: it monitors the
coefficient of variation (CV = std/mean) of timing measurements and stops once
results are statistically stable.

The algorithm:
1. Run at least `--min-iterations` (default: 10) before checking.
2. After each iteration, if CV < `--cv` threshold (default: 0.05), increment a stability counter; otherwise reset it.
3. Stop when the counter reaches `--stable-window` (default: 3) consecutive checks.
4. Never exceed `--max-iterations` (default: 200).

Use `-i N` to disable convergence and run exactly N iterations instead.

```bash
# Default: smart convergence (runs until CV < 5%)
pptree evaluate --simulate 1000x20x3 -t 50

# Stricter threshold with warmup
pptree evaluate --simulate 1000x20x3 -t 50 --warmup 2 --cv 0.03

# Tune convergence parameters
pptree evaluate --simulate 1000x20x3 -t 50 --min-iterations 20 --stable-window 5

# Fixed iterations (disables convergence)
pptree evaluate --simulate 1000x20x3 -t 50 -i 10
```

### Scenario Format

Scenarios are defined in JSON with shared defaults and per-scenario overrides:

```json
{
  "defaults": {
    "trees": 100, "lambda": 0.5, "vars": 0.5,
    "seed": 42, "warmup": 2,
    "convergence": { "cv_threshold": 0.05, "max_iterations": 200 }
  },
  "scenarios": [
    { "name": "small-forest",  "n": 200,  "p": 5,  "g": 2, "trees": 50 },
    { "name": "medium-forest", "n": 1000, "p": 20, "g": 3 },
    { "name": "fixed-5",       "n": 1000, "p": 20, "g": 3, "iterations": 5 }
  ]
}
```

Setting `"iterations"` forces fixed mode for that scenario; otherwise, smart convergence is used.

## R Package

### Building from Source

Install R dependencies, then build:

```bash
make r-install-deps
```

```bash
make r-build            # Prepare source and run R CMD build
make r-check            # Run R CMD check on the built tarball
make r-install          # Run R CMD INSTALL on the built tarball
make r-document         # Regenerate Roxygen man pages
make r-clean            # Remove compilation byproducts
```

> **Important:** Always use `make r-build` before checking or installing. This target copies the C++ core source into the R package's `src/core/` so it can be compiled on install.

### How `install_github` Works

R packages installed via `devtools::install_github()` must have the package source at the repository root. Since pptree is a monorepo, a CI workflow (`update-main-r.yml`) maintains a separate `main-r` branch for this purpose:

1. On every push to `main`, CI runs `make r-build` and `make r-untar` to produce the self-contained R package source
2. `git subtree split` extracts the package directory into the `main-r` branch
3. The `main-r` branch is force-pushed, keeping it in sync with `main`

This allows `devtools::install_github(..., ref = "main-r")` to work without requiring users to clone the full monorepo. A second workflow (`run-r-install-github.yml`) verifies that `install_github` succeeds on all three platforms.

## Development Tools

Install dependencies:

```bash
make install-tools      # Downloads uncrustify, cppcheck, doxygen into .tools/
make r-install-deps     # Installs R package dependencies (Rcpp, testthat, parsnip, etc.)
```

Run them:

```bash
make format             # Format C++ code (uncrustify)
make format-dry         # Check formatting without applying
make analyze            # Static analysis (cppcheck)
```

## Documentation

The project has a unified documentation site combining a static landing page, a C++ API reference (Doxygen), and R package documentation (pkgdown). The site is deployed to GitHub Pages with versioned directories for each branch and tag.

### Building Locally

```bash
make docs               # Build full site into docs/.build/
make docs-site          # Landing page only
make docs-cpp           # C++ API reference (Doxygen)
make docs-r             # R package site (pkgdown)
```

### Deployment

Documentation is automatically deployed to GitHub Pages on pushes to `main`, `next`, and version tags (`v*`). Each version gets its own directory:

```
/              Redirects to /main/
/main/         Latest from the main branch
/next/         Latest from the next branch
/v1.0.0/       Tagged release
```

GitHub Pages must be configured to deploy from the `gh-pages` branch (root).

## Reproducibility Break Protocol

This project guarantees that identical seeds produce identical results across
all supported platforms (Linux/GCC, macOS/Clang, Windows/MinGW). Golden files
in `golden/` are verified in CI on every platform.

If a code change intentionally alters model outputs for the same seed:

1. Open an issue or PR describing **why** the change is necessary
2. Regenerate golden files: `make golden-regen`
3. Verify all platforms pass: CI must be green on all three OS targets
4. Document the break in the PR description
5. Tag the release with a minor version bump

Implementation constraints that preserve reproducibility:
- **RNG**: pcg32 only (`stats::RNG`). Never `std::mt19937`.
- **Shuffling**: `stats::Uniform::distinct()` only. Never `std::shuffle`.
- **Sorting**: use `std::stable_sort` where element order affects downstream results. `std::sort` is not guaranteed to be stable and can produce different orderings of equal elements across platforms.
- **R seeds**: generated in R, passed as integers to C++.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
