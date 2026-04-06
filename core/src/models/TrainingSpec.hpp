#pragma once

#include "models/strategies/pp/PDA.hpp"
#include "models/strategies/vars/Uniform.hpp"
#include "models/strategies/vars/All.hpp"
#include "models/strategies/cutpoint/MeanOfMeans.hpp"
#include "models/strategies/stop/PureNode.hpp"
#include "models/strategies/binarize/LargestGap.hpp"
#include "models/strategies/partition/ByGroup.hpp"
#include "models/strategies/leaf/MajorityVote.hpp"

#include <memory>
#include <thread>
#include <nlohmann/json.hpp>

namespace ppforest2 {
  /**
   * @brief Training configuration for projection pursuit trees and forests.
   *
   * Composes seven strategies (projection pursuit, variable selection,
   * split cutpoint, stop rule, binarization, partition, leaf) together with
   * forest-level parameters (size, seed, threads, max retries).
   *
   * TrainingSpec is a concrete class — new strategies are plugged in
   * via the builder, not by subclassing:
   * @code
   *   // Single tree with PDA (lambda = 0.5):
   *   auto spec = TrainingSpec::builder()
   *       .pp(pp::pda(0.5))
   *       .build();
   *
   *   // Random forest with uniform variable selection:
   *   auto spec = TrainingSpec::builder()
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
    cutpoint::SplitCutpoint::Ptr const cutpoint;
    /** @brief Stop rule strategy. */
    stop::StopRule::Ptr const stop;
    /** @brief Binarization strategy. */
    binarize::Binarization::Ptr const binarize;
    /** @brief Partition strategy. */
    partition::StepPartition::Ptr const partition;
    /** @brief Leaf creation strategy. */
    leaf::LeafStrategy::Ptr const leaf;

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
     * All fields have sensible defaults. Call setters for only the
     * fields you want to customize, then build() or make().
     */
    struct Builder {
      pp::ProjectionPursuit::Ptr pp_           = pp::pda(0.0F);
      vars::VariableSelection::Ptr vars_       = vars::all();
      cutpoint::SplitCutpoint::Ptr cutpoint_   = cutpoint::mean_of_means();
      stop::StopRule::Ptr stop_                = stop::pure_node();
      binarize::Binarization::Ptr binarize_    = binarize::largest_gap();
      partition::StepPartition::Ptr partition_ = partition::by_group();
      leaf::LeafStrategy::Ptr leaf_            = leaf::majority_vote();

      int size_        = 0;
      int seed_        = 0;
      int threads_     = 0;
      int max_retries_ = 3;

      Builder& pp(pp::ProjectionPursuit::Ptr v) {
        pp_ = std::move(v);
        return *this;
      }
      Builder& vars(vars::VariableSelection::Ptr v) {
        vars_ = std::move(v);
        return *this;
      }
      Builder& cutpoint(cutpoint::SplitCutpoint::Ptr v) {
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
      Builder& partition(partition::StepPartition::Ptr v) {
        partition_ = std::move(v);
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

      TrainingSpec build() {
        return TrainingSpec(
            std::move(pp_),
            std::move(vars_),
            std::move(cutpoint_),
            std::move(stop_),
            std::move(binarize_),
            std::move(partition_),
            std::move(leaf_),
            size_,
            seed_,
            threads_,
            max_retries_
        );
      }

      Ptr make() { return std::make_shared<TrainingSpec>(build()); }
    };

    /** @brief Create a builder with all defaults. */
    static Builder builder() { return Builder{}; }

    /**
     * @brief Construct a training specification.
     *
     * @param pp           Projection pursuit strategy.
     * @param vars         Variable selection strategy.
     * @param cutpoint     Split cutpoint strategy.
     * @param stop         Stop rule strategy.
     * @param binarize     Binarization strategy.
     * @param partition    Partition strategy.
     * @param leaf         Leaf creation strategy.
     * @param size         Number of trees (0 = single tree).
     * @param seed         RNG seed.
     * @param threads      Number of threads (0 = hardware concurrency).
     * @param max_retries  Maximum retry attempts for degenerate trees.
     */
    TrainingSpec(
        pp::ProjectionPursuit::Ptr pp,
        vars::VariableSelection::Ptr vars,
        cutpoint::SplitCutpoint::Ptr cutpoint,
        stop::StopRule::Ptr stop,
        binarize::Binarization::Ptr binarize,
        partition::StepPartition::Ptr partition,
        leaf::LeafStrategy::Ptr leaf,
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
    partition::StepPartition::Result split(NodeContext& ctx, stats::RNG& rng) const { return (*partition)(ctx, rng); }

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
