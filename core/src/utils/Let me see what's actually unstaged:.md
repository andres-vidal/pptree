Let me see what's actually unstaged:

179 files, roughly three independent themes stacked on top of each other. Reviewing in the order they landed would be confusing; reviewing top-down by layer is what you want. Proposed order:

<!-- ## Tier 1 — Build system & structural reorg (~10 files)

Start here — sets the vocabulary for everything else. If these make sense, the names used below are unambiguous.

1. `core/src/models/strategies/Strategy.hpp` — the `primary_param_registry()` + macro additions. The shape of the strategy framework.
2. `core/src/models/strategies/CMakeLists.txt` (new) + `core/src/models/CMakeLists.txt` + `core/CMakeLists.txt` — the `ppforest2-strategies` / `ppforest2-core` split, `$<LINK_LIBRARY:WHOLE_ARCHIVE,…>`, `cmake_minimum_required(VERSION 3.24)`.
3. `core/src/models/TrainingSpec.cpp` — mainly notable for what's *gone* (force-link block).
4. `bindings/R/configure` + `bindings/R/src/Makevars.in`/`.win` + top-level `Makefile` — the R-side mirror of the library split.
5. `data/` reorg (`classification/`, `regression/`), `CLI.integration.hpp` path constants, `IO.test.cpp`, `GoldenGen.cpp`, `Reproducibility.test.cpp` — same mechanical rename, different files. Skim.

## Tier 2 — Type system & unified `y` (~4 files)

Foundations that every regression diff assumes.

1. `core/src/utils/Types.hpp` — `Mode` enum, `Outcome` / `GroupId` distinction.
2. `core/src/models/TreeBranch.hpp` — the `Cutpoint` alias drop (trivial, read with Types.hpp).

## Tier 3 — Regression support (biggest chunk, ~40 files) -->

The heart of the change. Review in this sub-order:

1. **Model hierarchy (new virtual polymorphism):**
   - `core/src/models/Tree.hpp`/`cpp` + `ClassificationTree.{hpp,cpp}` + `RegressionTree.{hpp,cpp}`.
   - `core/src/models/Forest.hpp`/`cpp` + `ClassificationForest.{hpp,cpp}` + `RegressionForest.{hpp,cpp}`.
   - `core/src/models/Bagged.hpp` (replaces `BootstrapTree` hierarchy).
   - `core/src/models/TrainingSpec.hpp`/`cpp` (mode enum, strategy validation via `supported_modes()`).
2. **New / renamed strategies (fastest to review — each is small and self-contained):**
   - `strategies/cutpoint/Cutpoint.hpp` (renamed from `SplitCutpoint`).
   - `strategies/grouping/` (new family: `Grouping.hpp`, `ByLabel.{hpp,cpp}`, `ByCutpoint.{hpp,cpp}`).
   - `strategies/leaf/MeanResponse.{hpp,cpp}` (new).
   - `strategies/stop/{MinSize, MinVariance, MaxDepth, CompositeStop}.{hpp,cpp}` (new).
3. **Regression stats + serialization:**
   - `core/src/stats/RegressionMetrics.{hpp,cpp}`.
   - `core/src/stats/Simulation.cpp` — the regression-split fix.
   - `core/src/serialization/Json.{hpp,cpp}` + `JsonOptional.hpp` (null-instead-of-absent convention, `has_value` helper).
   - `core/src/serialization/ExportValidation.{hpp,cpp}` (new validation pass).
4. **I/O:** `core/src/io/IO.{hpp,cpp}` — `read_regression_sorted`.

## Tier 4 — CLI layer (~10 files)

Now you've seen the core, these read as "expose what's in Tier 3":

1. `cli/ModelParams.hpp`/`cpp` — mode parameter, `stop_inputs: vector<string>`, `strategy_string_to_json` with positional shorthand.
2. `cli/Train.cpp` — `--mode`, repeatable `--stop`.
3. `cli/Predict.cpp`, `Evaluate.cpp`, `Summarize.cpp` — mode-aware output paths.

## Tier 5 — R bindings (~20 files)

Mirror of Tier 3 on the R side. Pairs cleanly:

1. `bindings/R/R/util.R` (validate_data + mode auto-detect + NA rejection), `R/ppmodel.R` (new, shared S3 scaffolding + lazy OOB accessors).
2. `R/pptr.R`, `R/pprf.R` — mode-branched paths, new S3 class vectors.
3. `R/{pp,vars,stop,leaf,partition}-strategy.R` — wrappers; `stop-strategy.R` and `partition-strategy.R` (renamed to grouping family) see the most change.
4. `R/data.R` — iris block removed, california_housing added.
5. `R/parsnip.R` — regression mode registration.
6. `bindings/R/inst/include/ppforest2.h` + `bindings/R/src/main.cpp` — Rcpp conversions + new exports.

## Tier 6 — Tests (~25 files)

By this point the tests read as "here's what we promised in Tiers 2–5 being enforced." Spot-check a few rather than reading all line-by-line:

- `core/src/models/Regression.test.cpp` (new) — paired with Tier 3 hierarchy.
- `core/src/cli/Train.integration.test.cpp` — new `TrainPredictRegressionMtcars`, `TrainStopRepeated*`, `TrainStopCLI*Overrides*`.
- `core/src/cli/ModelParams.test.cpp` — positional-shorthand block.
- `bindings/R/tests/testthat/test-regression.R` (new) — end-to-end R regression round-trip.
- `bindings/R/tests/testthat/test-reproducibility.R` — `prepare_data(..., response = ...)` parameter for iris.
- Plot snapshots: `_snaps/plot-{structure,importance}/pprf-regression-*.svg` (new, auto-generated).

## Tier 7 — Docs, benchmarks, release plumbing (~15 files)

Skim-only:

- `CHANGELOG.md` — the user-facing summary; sanity-check against what you've seen in Tiers 3-5.
- `README.md`, `bindings/R/vignettes/*.Rmd`, `mainpage.dox`, `extending-strategies.dox` — iris renames, benchmark path updates, regression caveats.
- `bench/default-scenarios-{classification,regression}.json` — scenarios split.
- `.github/workflows/release.yml`, `run-benchmark.yml`, `dependabot.yml` — CI hardening (SHA pinning, two-family benchmark tables, CRAN check, tag-from-main, VERSION/DESCRIPTION consistency).

## Reviewing discipline

- **After each tier, run the relevant test subset.** `make test` after Tiers 1–3, `make r-test` after Tier 5. The green signal between tiers prevents compounding confusion.
- **Don't read every CMake diff character-by-character** — most are boilerplate mass-renames (`data/foo.csv` → `data/classification/foo.csv`, `ppforest2-models` → `ppforest2-strategies`). Skim for anomalies; trust the fact that tests pass.
- **Read `.hpp` before `.cpp`** within each file pair — the header is the interface you're actually being asked to approve; the `.cpp` is implementation.
- **Open CHANGELOG last, not first.** If you read the changelog first, you go into review looking for what it promised. If you read it last, it's a cross-check that what you just approved matches the user-facing narrative.

This puts ~50 files in the first four tiers (the substantive review), ~50 in tests/docs (skim), and 80 in mechanical renames you can trust.