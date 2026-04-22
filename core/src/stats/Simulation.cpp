#include "stats/Simulation.hpp"
#include "stats/Normal.hpp"
#include "stats/Uniform.hpp"
#include "stats/GroupPartition.hpp"

#include <algorithm>
#include <numeric>
#include <string>

namespace ppforest2::stats {
  DataPacket simulate(int const n, int const p, int const G, RNG& rng, SimulationParams const& params) {
    types::FeatureMatrix x(n, p);
    types::GroupIdVector y_int(n);

    for (int i = 0; i < n; ++i) {
      types::Feature group_mean = params.mean + (i % G) * params.mean_separation;

      Normal norm(group_mean, params.sd);

      for (int j = 0; j < p; ++j) {
        x(i, j) = norm(rng);
      }

      y_int[i] = i % G;
    }

    sort(x, y_int);

    types::OutcomeVector y = y_int.cast<types::Outcome>();

    // Synthesize placeholder group names so the returned DataPacket upholds
    // the classification invariant "group_names has one entry per class."
    // Without this, `group_names` would be `[]`, which cascades into
    // out-of-bounds reads in labeled serialization paths (see
    // `Export<Model::Ptr>::to_json`). Real-data readers (e.g. CSV) populate
    // names from the label column — simulation has no source, so we fall
    // back to the integer index as a string.
    std::vector<std::string> group_names;
    group_names.reserve(static_cast<std::size_t>(G));
    for (int g = 0; g < G; ++g) {
      group_names.emplace_back(std::to_string(g));
    }

    return DataPacket(x, y, /*group_names=*/group_names, /*feature_names=*/{});
  }

  Split split(DataPacket const& data, float train_ratio, RNG& rng) {
    int const n          = data.x.rows();
    int const train_size = static_cast<int>(n * train_ratio);

    std::vector<int> train_indices;
    std::vector<int> test_indices;

    train_indices.reserve(train_size);
    test_indices.reserve(n - train_size);

    if (data.groups.empty()) {
      // Regression path: `data.groups` is empty by construction (no
      // discrete class labels to stratify on). Draw a uniform random
      // subset of row indices, then sort each side so that callers
      // which rely on y-sorted contiguity (e.g. `ByCutpoint::init`
      // at the root of the regression training pipeline) still see
      // the correct shape.
      Uniform unif(0, n - 1);
      std::vector<int> indices = unif.distinct(n, rng);

      std::vector<int> tr(indices.begin(), indices.begin() + train_size);
      std::vector<int> te(indices.begin() + train_size, indices.end());

      // Preserve the y-sorted row ordering on each side. `data.y` is
      // already sorted ascending (regression DataPackets are produced
      // that way by `io::csv::read_regression_sorted` and
      // `simulate_regression`), so sorting by index is equivalent to
      // sorting by y.
      std::sort(tr.begin(), tr.end());
      std::sort(te.begin(), te.end());

      return {.tr = std::move(tr), .te = std::move(te)};
    }

    // Classification path: stratified per-group split to preserve class
    // balance in both sides.
    GroupPartition spec(data.y);

    for (auto const& group : data.groups) {
      int group_start      = spec.group_start(group);
      int group_size       = spec.group_size(group);
      int group_end        = group_start + group_size - 1;
      int group_train_size = static_cast<int>(group_size * train_ratio);

      Uniform unif(group_start, group_end);
      std::vector<int> group_indices = unif.distinct(group_size, rng);

      train_indices.insert(train_indices.end(), group_indices.begin(), group_indices.begin() + group_train_size);
      test_indices.insert(test_indices.end(), group_indices.begin() + group_train_size, group_indices.end());
    }

    return {.tr = train_indices, .te = test_indices};
  }

  DataPacket simulate_regression(int const n, int const p, RNG& rng, RegressionSimulationParams const& params) {
    using namespace types;

    int const n_informative = params.n_informative > 0 ? std::min(params.n_informative, p) : std::min(p, 5);

    // Generate features from N(0, feature_sd^2).
    FeatureMatrix x(n, p);
    Normal feature_dist(Feature(0), params.feature_sd);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < p; ++j) {
        x(i, j) = feature_dist(rng);
      }
    }

    // Generate fixed coefficients for the first n_informative features.
    // Use deterministic values so different runs with the same n_informative
    // produce comparable datasets.
    std::vector<Feature> coef(static_cast<std::size_t>(n_informative));

    for (int j = 0; j < n_informative; ++j) {
      coef[static_cast<std::size_t>(j)] = static_cast<Feature>(j + 1);
    }

    // Linear response + noise.
    Normal noise_dist(Feature(0), params.noise_sd);
    OutcomeVector y_cont(n);

    for (int i = 0; i < n; ++i) {
      Feature y = params.intercept;

      for (int j = 0; j < n_informative; ++j) {
        y += coef[static_cast<std::size_t>(j)] * x(i, j);
      }

      y += noise_dist(rng);
      y_cont(i) = y;
    }

    // Sort rows by the continuous response.
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);

    std::stable_sort(order.begin(), order.end(), [&y_cont](int a, int b) { return y_cont(a) < y_cont(b); });

    FeatureMatrix sorted_x(n, p);
    OutcomeVector sorted_y(n);

    for (int i = 0; i < n; ++i) {
      sorted_x.row(i) = x.row(order[static_cast<std::size_t>(i)]);
      sorted_y(i)     = y_cont(order[static_cast<std::size_t>(i)]);
    }

    return DataPacket(sorted_x, sorted_y, DataPacket::NoGroups{});
  }
}
