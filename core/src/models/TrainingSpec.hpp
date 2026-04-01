#pragma once

#include "models/PPPDAStrategy.hpp"
#include "models/DRUniformStrategy.hpp"
#include "models/DRNoopStrategy.hpp"
#include "models/SRMeanOfMeansStrategy.hpp"

#include <memory>
#include <thread>
#include <nlohmann/json.hpp>

namespace ppforest2 {
  /**
   * @brief Training configuration for projection pursuit trees and forests.
   *
   * Composes a projection pursuit strategy (PPStrategy), a
   * dimensionality reduction strategy (DRStrategy), and a split
   * strategy (SRStrategy), together with forest-level parameters
   * (size, seed, threads, max retries).
   *
   * TrainingSpec is a concrete class — new strategies are plugged in
   * via the constructor, not by subclassing:
   * @code
   *   // Single tree with PDA:
   *   TrainingSpec spec(pp::pda(0.5), dr::noop(), sr::mean_of_means());
   *
   *   // Random forest with uniform variable selection:
   *   TrainingSpec spec(pp::pda(0.0), dr::uniform(3), sr::mean_of_means(),
   *                     100, 0);  // size, seed
   *
   *   // Custom strategy:
   *   TrainingSpec spec(my_custom_pp(), dr::uniform(5), sr::mean_of_means(),
   *                     200, 7);  // size, seed
   * @endcode
   *
   * Strategies are held via shared_ptr and are immutable after
   * construction, so TrainingSpec can be freely copied and shared
   * across trees without deep cloning.
   */
  struct TrainingSpec {
    using Ptr = std::shared_ptr<TrainingSpec>;

    /** @brief Projection pursuit optimization strategy. */
    pp::PPStrategy::Ptr const pp_strategy;
    /** @brief Dimensionality reduction strategy. */
    dr::DRStrategy::Ptr const dr_strategy;
    /** @brief Group splitting rule strategy. */
    sr::SRStrategy::Ptr const sr_strategy;

    /** @brief Number of trees (0 = single tree). */
    int const size;
    /** @brief RNG seed. */
    int const seed;
    /** @brief Number of threads for parallel forest training. */
    int const threads;
    /** @brief Maximum retry attempts for degenerate trees. */
    int const max_retries;

    /**
     * @brief Construct a training specification.
     *
     * @param pp           Projection pursuit strategy.
     * @param dr           Dimensionality reduction strategy.
     * @param sr           Split rule strategy.
     * @param size         Number of trees (0 = single tree).
     * @param seed         RNG seed.
     * @param threads    Number of threads (0 = hardware concurrency).
     * @param max_retries  Maximum retry attempts for degenerate trees.
     */
    TrainingSpec(pp::PPStrategy::Ptr pp,
                 dr::DRStrategy::Ptr dr,
                 sr::SRStrategy::Ptr sr,
                 int size        = 0,
                 int seed        = 0,
                 int threads     = 0,
                 int max_retries = 3);

    /** @brief Whether this specification describes a forest (size > 0). */
    bool is_forest() const { return size > 0; }

    /** @brief Serialize the training spec to JSON. */
    void to_json(nlohmann::json& j) const;

    /** @brief Deserialize a training spec from JSON. */
    static Ptr from_json(nlohmann::json const& j);

    /**
     * @brief Create a shared pointer to a TrainingSpec.
     *
     * Convenience factory that forwards all arguments to the constructor.
     *
     * @return Shared pointer to a new TrainingSpec.
     */
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
