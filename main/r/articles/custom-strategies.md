# Custom strategies

ppforest2 trains trees by composing three pluggable strategies:

| Strategy                          | Purpose                                            | Built-in                                                                                                                                                                       |
|-----------------------------------|----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **PP** (projection pursuit)       | Find the projection that best separates groups     | [`pp_pda()`](https://andres-vidal.github.io/ppforest2/main/r/reference/pp_pda.md) — Penalized Discriminant Analysis                                                            |
| **DR** (dimensionality reduction) | Select which variables are available at each split | [`dr_uniform()`](https://andres-vidal.github.io/ppforest2/main/r/reference/dr_uniform.md), [`dr_noop()`](https://andres-vidal.github.io/ppforest2/main/r/reference/dr_noop.md) |
| **SR** (split rule)               | Compute the split threshold in projected space     | [`sr_mean_of_means()`](https://andres-vidal.github.io/ppforest2/main/r/reference/sr_mean_of_means.md)                                                                          |

You can add new strategies without modifying the core tree-building
logic. This vignette walks through the process.

## How strategies work

Each strategy is an R list with a `name` field that identifies it, a
`display_name` for summaries, and any parameters the strategy needs. The
`name` must match a C++ strategy registered under the same name.

``` r
library(ppforest2)
#> 
#> Attaching package: 'ppforest2'
#> The following object is masked from 'package:datasets':
#> 
#>     iris

pp_pda(0.5)
#> $name
#> [1] "pda"
#> 
#> $display_name
#> [1] "PDA"
#> 
#> $lambda
#> [1] 0.5
#> 
#> attr(,"class")
#> [1] "pp_strategy"
dr_uniform(n_vars = 2)
#> $name
#> [1] "uniform"
#> 
#> $display_name
#> [1] "Uniform random"
#> 
#> $n_vars
#> [1] 2
#> 
#> $p_vars
#> NULL
#> 
#> attr(,"class")
#> [1] "dr_strategy"
sr_mean_of_means()
#> $name
#> [1] "mean_of_means"
#> 
#> $display_name
#> [1] "Mean of means"
#> 
#> attr(,"class")
#> [1] "sr_strategy"
```

When you call
[`pptr()`](https://andres-vidal.github.io/ppforest2/main/r/reference/pptr.md)
or
[`pprf()`](https://andres-vidal.github.io/ppforest2/main/r/reference/pprf.md),
the strategy lists are passed to C++, where the `name` field dispatches
to the corresponding C++ implementation. The actual computation
(optimization, variable selection, threshold) happens entirely in C++.

## Adding a new strategy

Adding a strategy requires work on both sides:

1.  **C++**: Implement the strategy class (the computation).
2.  **R**: Write a constructor function (the user-facing API).

### Step 1: C++ implementation

Each strategy family has a base class with pure virtual methods. Your
new strategy inherits from the appropriate base and implements them.

For example, a new projection pursuit strategy needs to implement
`index()` (evaluate a projection) and
[`optimize()`](https://rdrr.io/r/stats/optimize.html) (find the best
projection):

``` cpp
// File: core/src/models/PPMyStrategy.hpp
#pragma once
#include "models/PPStrategy.hpp"
#include "models/Strategy.hpp"
#include "utils/JsonValidation.hpp"

namespace ppforest2::pp {

struct PPMyStrategy : public PPStrategy {
  explicit PPMyStrategy(float alpha) : alpha_(alpha) {}

  std::string display_name() const override { return "My method"; }

  types::Feature index(
    const types::FeatureMatrix&  x,
    const stats::GroupPartition& group_spec,
    const Projector&             projector) const override {
    // Evaluate how good this projection is.
    // Return a scalar index (higher = better separation).
    ...
  }

  PPResult optimize(
    const types::FeatureMatrix&  x,
    const stats::GroupPartition& group_spec) const override {
    // Find the optimal projector for the data.
    // Return PPResult{ projector_vector, index_value }.
    ...
  }

  void to_json(nlohmann::json& j) const override {
    j = {{"name", "my_method"}, {"alpha", alpha_}};
  }

  static PPStrategy::Ptr from_json(const nlohmann::json& j) {
    validate_json_keys(j, "my_method PP", {"name", "alpha"});
    return my_method(j.at("alpha").get<float>());
  }

  PPFOREST2_REGISTER_STRATEGY(PPStrategy, "my_method")

private:
  const float alpha_;
};

inline PPStrategy::Ptr my_method(float alpha) {
  return std::make_shared<PPMyStrategy>(alpha);
}

}  // namespace ppforest2::pp
```

The key pieces:

- **`to_json()`** serializes the strategy name and parameters. This is
  used for model persistence.
- **`from_json()`** deserializes from JSON and validates that no
  unexpected keys are present.
- **`PPFOREST2_REGISTER_STRATEGY`** registers the factory so JSON
  deserialization finds it automatically.
- **`display_name()`** returns a human-readable label for summaries.
- **Factory function** (`my_method()`) is a convenience wrapper.

The same pattern applies to DR strategies (`select()`) and SR strategies
(`threshold()`). See the C++ documentation (`extending-strategies.dox`)
for complete interface definitions and examples for all three families.

After writing the `.cpp` file, add it to
`core/src/models/CMakeLists.txt`.

### Step 2: R constructor

Write an R function that creates a strategy list. The `name` field must
match the C++ registration name exactly.

``` r
#' My custom projection pursuit strategy.
#'
#' @param alpha A tuning parameter.
#' @return A \code{pp_strategy} object.
#' @export
pp_my_method <- function(alpha = 1.0) {
  if (!is.numeric(alpha) || length(alpha) != 1)
    stop("`alpha` must be a single number.")

  structure(
    list(name = "my_method", display_name = "My method", alpha = alpha),
    class = "pp_strategy"
  )
}
```

The constructor should:

- **Validate parameters** before they reach C++. Catching errors early
  with clear messages is better than a C++ exception.
- **Set the S3 class** to `pp_strategy`, `dr_strategy`, or
  `sr_strategy`. This is checked by `resolve_strategies()`.
- **Include `display_name`** for readable output in
  [`summary()`](https://rdrr.io/r/base/summary.html).
- **Use the same parameter names** as `to_json()` in C++. The R list is
  converted to JSON and passed to `from_json()` on the C++ side.

### Step 3: Use it

Once both sides are in place, the new strategy works like any built-in:

``` r
# Single tree
tree <- pptr(Type ~ ., data = iris, pp = pp_my_method(alpha = 0.5))

# Forest
forest <- pprf(Type ~ ., data = iris, pp = pp_my_method(alpha = 0.5), dr = dr_uniform(2))

# Summary shows the strategy
summary(tree)
```

The strategy is also available from the CLI:

``` bash
ppforest2 train -d iris.csv --pp my_method:alpha=0.5
```

And models trained with the new strategy can be saved and loaded as
usual — the JSON registry handles serialization automatically.

## Strategy families reference

### PP: Projection pursuit

Controls how the tree finds the best linear combination of variables at
each node.

    index(x, group_spec, projector) -> scalar
    optimize(x, group_spec) -> PPResult{projector, index}

[`optimize()`](https://rdrr.io/r/stats/optimize.html) is the main
method. It receives the data matrix and group partition and returns the
best projection vector. `index()` evaluates a given projection (used for
variable importance calculations).

### DR: Dimensionality reduction

Controls which variables are available to projection pursuit at each
split. This is what makes random forests “random”.

    select(x, group_spec, rng) -> DRResult{selected_indices, original_cols}

The returned `DRResult` tracks which columns were selected so the
reduced-space projector can be expanded back to the full feature space.

### SR: Split rule

Controls where the split threshold is placed in the projected space.

    threshold(group_1, group_2, projector) -> scalar

Receives the two groups (already partitioned by projection pursuit) and
the projection vector. Returns the threshold value.

## Checklist

1.  Create `core/src/models/MyStrategy.hpp` (and `.cpp` if needed).
2.  Inherit from `PPStrategy`, `DRStrategy`, or `SRStrategy`.
3.  Implement the pure virtual methods.
4.  Implement `to_json()` with a `"name"` field.
5.  Implement `display_name()` for human-readable summaries.
6.  Add `static Ptr from_json()` with key validation.
7.  Add `PPFOREST2_REGISTER_STRATEGY(Base, "name")`.
8.  Add a factory function in the strategy’s namespace.
9.  Add the `.cpp` to `core/src/models/CMakeLists.txt`.
10. Write tests in `MyStrategy.test.cpp` (JSON round-trip + functional).
11. Write the R constructor function with validation, `display_name`,
    and the correct S3 class.
12. Export and document the R function.
