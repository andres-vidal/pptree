#pragma once

#include "models/strategies/pp/ProjectionPursuit.hpp"
#include "models/strategies/Strategy.hpp"
#include "stats/GroupPartition.hpp"
#include "utils/Types.hpp"

namespace ppforest2::pp {
  /**
   * @brief Penalized Discriminant Analysis projection pursuit strategy.
   *
   * Optimizes a linear discriminant projection using a penalized
   * between-group / within-group variance ratio. The @c lambda
   * parameter controls the penalty strength in the LDA index.
   */
  struct PDA : public ProjectionPursuit {
    explicit PDA(float lambda);

    nlohmann::json to_json() const override;
    std::string display_name() const override { return lambda == 0 ? "LDA" : "PDA"; }
    std::set<types::Mode> supported_modes() const override {
      return {types::Mode::Classification, types::Mode::Regression};
    }

    /**
     * @brief NodeContext-based interface: optimize projection and write to ctx.
     *
     * Reads ctx.x, ctx.var_selection, ctx.active_partition(). Writes ctx.projector, ctx.pp_index_value.
     */
    void optimize(NodeContext& ctx, stats::RNG& rng) const override;

    /**
     * @brief Direct computation: find optimal projection for given data and partition.
     *
     * This is the core PDA/LDA optimization logic, usable independently of NodeContext.
     */
    ProjectionPursuit::Result compute(types::FeatureMatrix const& x, stats::GroupPartition const& y_part) const;

    static ProjectionPursuit::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(ProjectionPursuit, "pda")
    PPFOREST2_REGISTER_PRIMARY_PARAM("pda", "lambda")

  private:
    /** @brief Penalty parameter for the LDA index (0 = standard LDA). */
    float const lambda;
  };
}
