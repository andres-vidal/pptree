#pragma once

#include "models/strategies/pp/ProjectionPursuit.hpp"
#include "models/strategies/vars/VariableSelection.hpp"
#include "models/strategies/cutpoint/Cutpoint.hpp"
#include "models/strategies/stop/StopRule.hpp"
#include "models/strategies/binarize/Binarization.hpp"
#include "models/strategies/grouping/Grouping.hpp"
#include "models/strategies/leaf/LeafStrategy.hpp"

#include <memory>
#include <thread>
#include <nlohmann/json.hpp>

namespace ppforest2 {
  /**
   * @brief Training configuration for projection pursuit trees and forests.
   *
   * Composes seven strategies (projection pursuit, variable selection,
   * split cutpoint, stop rule, binarization, grouping, leaf) together with
   * forest-level parameters (size, seed, threads, max retries).
   *
   * TrainingSpec is a concrete class — new strategies are plugged in
   * via the builder, not by subclassing:
   * @code
   *   // Single tree with PDA (lambda = 0.5):
   *   auto spec = TrainingSpec::builder(types::Mode::Classification)
   *       .pp(pp::pda(0.5))
   *       .build();
   *
   *   // Random forest with uniform variable selection:
   *   auto spec = TrainingSpec::builder(types::Mode::Classification)
   *       .size(100)
   *       .pp(pp::pda(0.0))
   *       .vars(vars::uniform(3))
   *       .build();
   * @endcode
   *
   * Strategies are held via shared_ptr and are immutable after
   * construction, so TrainingSpec can be freely copied and shared
   * across trees without deep cloning.
   */
  struct TrainingSpec {
    using Ptr = std::shared_ptr<TrainingSpec>;

    /** @brief Projection pursuit optimization strategy. */
    pp::ProjectionPursuit::Ptr const pp;
    /** @brief Variable selection strategy. */
    vars::VariableSelection::Ptr const vars;
    /** @brief Split cutpoint strategy. */
    cutpoint::Cutpoint::Ptr const cutpoint;
    /** @brief Stop rule strategy. */
    stop::StopRule::Ptr const stop;
    /** @brief Binarization strategy. */
    binarize::Binarization::Ptr const binarize;
    /** @brief Grouping strategy. */
    grouping::Grouping::Ptr const grouping;
    /** @brief Leaf creation strategy. */
    leaf::LeafStrategy::Ptr const leaf;

    /** @brief Training mode (classification or regression). */
    types::Mode const mode;

    /** @brief Number of trees (0 = single tree). */
    int const size;
    /** @brief RNG seed. */
    int const seed;
    /** @brief Number of threads for parallel forest training. */
    int const threads;
    /** @brief Maximum retry attempts for degenerate trees. */
    int const max_retries;

    /**
     * @brief Fluent builder for TrainingSpec.
     *
     * `mode` is required at construction — it's structurally primary
     * (drives the default binarize strategy and gates mode-compatibility
     * checks), not a post-construction tweak, so there's no `.mode()`
     * setter. Other strategies default to mode-agnostic factory values
     * (`pp::pda`, `vars::all`, etc.) that callers override via setters.
     * `binarize` defaults lazily at `build()` time: `largest_gap` for
     * classification, `disabled` for regression.
     *
     * Default initialisation of the strategy fields is done in the
     * constructor definition in `TrainingSpec.cpp` so the header can
     * include only base-class strategy headers.
     */
    struct Builder {
      pp::ProjectionPursuit::Ptr pp_;
      vars::VariableSelection::Ptr vars_;
      cutpoint::Cutpoint::Ptr cutpoint_;
      stop::StopRule::Ptr stop_;
      binarize::Binarization::Ptr binarize_ = nullptr;
      grouping::Grouping::Ptr grouping_;
      leaf::LeafStrategy::Ptr leaf_;

      types::Mode const mode_;

      int size_        = 0;
      int seed_        = 0;
      int threads_     = 0;
      int max_retries_ = 3;

      explicit Builder(types::Mode mode);

      Builder& pp(pp::ProjectionPursuit::Ptr v) {
        pp_ = std::move(v);
        return *this;
      }
      Builder& vars(vars::VariableSelection::Ptr v) {
        vars_ = std::move(v);
        return *this;
      }
      Builder& cutpoint(cutpoint::Cutpoint::Ptr v) {
        cutpoint_ = std::move(v);
        return *this;
      }
      Builder& stop(stop::StopRule::Ptr v) {
        stop_ = std::move(v);
        return *this;
      }
      Builder& binarize(binarize::Binarization::Ptr v) {
        binarize_ = std::move(v);
        return *this;
      }
      Builder& grouping(grouping::Grouping::Ptr v) {
        grouping_ = std::move(v);
        return *this;
      }
      Builder& leaf(leaf::LeafStrategy::Ptr v) {
        leaf_ = std::move(v);
        return *this;
      }

      Builder& size(int v) {
        size_ = v;
        return *this;
      }
      Builder& seed(int v) {
        seed_ = v;
        return *this;
      }
      Builder& threads(int v) {
        threads_ = v;
        return *this;
      }
      Builder& max_retries(int v) {
        max_retries_ = v;
        return *this;
      }

      /**
       * @brief Finalize the builder into a `TrainingSpec`.
       *
       * Resolves the mode-dependent default for `binarize` lazily:
       * `largest_gap` for classification, `disabled` for regression.
       * Defined in `TrainingSpec.cpp` so the factory-call concrete
       * headers stay out of this interface.
       */
      TrainingSpec build();

      /** @brief Shorthand for `std::make_shared<TrainingSpec>(build())`. */
      Ptr make();
    };

    /**
     * @brief Create a builder for the given mode.
     *
     * Mode is required because it's structurally primary: it determines
     * the default `binarize` strategy and the mode-compatibility checks
     * for every other strategy. There is no "no-mode" TrainingSpec.
     */
    static Builder builder(types::Mode mode) { return Builder{mode}; }

    /**
     * @brief Construct a training specification.
     *
     * @param pp           Projection pursuit strategy.
     * @param vars         Variable selection strategy.
     * @param cutpoint     Split cutpoint strategy.
     * @param stop         Stop rule strategy.
     * @param binarize     Binarization strategy.
     * @param grouping     Grouping strategy.
     * @param leaf         Leaf creation strategy.
     * @param size         Number of trees (0 = single tree).
     * @param seed         RNG seed.
     * @param threads      Number of threads (0 = hardware concurrency).
     * @param max_retries  Maximum retry attempts for degenerate trees.
     */
    TrainingSpec(
        pp::ProjectionPursuit::Ptr pp,
        vars::VariableSelection::Ptr vars,
        cutpoint::Cutpoint::Ptr cutpoint,
        stop::StopRule::Ptr stop,
        binarize::Binarization::Ptr binarize,
        grouping::Grouping::Ptr grouping,
        leaf::LeafStrategy::Ptr leaf,
        types::Mode mode,
        int size,
        int seed,
        int threads,
        int max_retries
    );

    // -- Forwarding methods (delegate to the underlying strategy) -----------

    /** @brief Run projection pursuit optimization. */
    void find_projection(NodeContext& ctx, stats::RNG& rng) const { (*pp)(ctx, rng); }

    /** @brief Run variable selection. */
    void select_vars(NodeContext& ctx, stats::RNG& rng) const { (*vars)(ctx, rng); }

    /** @brief Compute the split cutpoint. */
    void find_cutpoint(NodeContext& ctx, stats::RNG& rng) const { (*cutpoint)(ctx, rng); }

    /** @brief Check whether the node should stop growing. */
    bool should_stop(NodeContext const& ctx, stats::RNG& rng) const { return (*stop)(ctx, rng); }

    /** @brief Reduce multiclass partition to binary. */
    void regroup(NodeContext& ctx, stats::RNG& rng) const { (*binarize)(ctx, rng); }

    /** @brief Split observations into two child partitions. */
    grouping::Grouping::Result group(NodeContext& ctx, stats::RNG& rng) const { return (*grouping)(ctx, rng); }

    /** @brief Create the initial group partition from the training response. */
    stats::GroupPartition init_groups(types::OutcomeVector const& y) const { return grouping->init(y); }

    /** @brief Create a leaf node from the current node context. */
    TreeNode::Ptr create_leaf(NodeContext const& ctx, stats::RNG& rng) const { return (*leaf)(ctx, rng); }

    /** @brief Whether this specification describes a forest (size > 0). */
    bool is_forest() const { return size > 0; }

    /** @brief Serialize the training spec to JSON. */
    nlohmann::json to_json() const;

    /** @brief Deserialize a training spec from JSON. */
    static Ptr from_json(nlohmann::json const& j);

    /** @brief Create a shared pointer to a TrainingSpec. */
    template<typename... Args> static Ptr make(Args&&... args) {
      return std::make_shared<TrainingSpec>(std::forward<Args>(args)...);
    }

    /** @brief
     * Get the number of threads to use for training.
     *
     * If the number of threads is not specified, the number of hardware
     * concurrency is returned.
     *
     * @return The number of threads to use for training.
     */
    int resolve_threads() const { return threads > 0 ? threads : std::thread::hardware_concurrency(); }
  };
}
