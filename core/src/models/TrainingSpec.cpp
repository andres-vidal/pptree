#include "models/TrainingSpec.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"
#include "utils/UserError.hpp"

#include <string>

namespace ppforest2 {
  // -- Builder -------------------------------------------------------------
  // Defaults and finalize logic out-of-line so `TrainingSpec.hpp` stays
  // on base-class strategy headers and skips the concrete-strategy
  // includes needed only to make factory calls below.

  TrainingSpec::Builder::Builder(types::Mode mode)
      : pp_(pp::pda(0.0F))
      , vars_(vars::all())
      , cutpoint_(cutpoint::mean_of_means())
      , stop_(stop::pure_node())
      , grouping_(grouping::by_label())
      , leaf_(leaf::majority_vote())
      , mode_(mode) {}

  TrainingSpec TrainingSpec::Builder::build() {
    if (!binarize_) {
      binarize_ = mode_ == types::Mode::Regression ? binarize::disabled() : binarize::largest_gap();
    }
    return TrainingSpec(
        std::move(pp_),
        std::move(vars_),
        std::move(cutpoint_),
        std::move(stop_),
        std::move(binarize_),
        std::move(grouping_),
        std::move(leaf_),
        mode_,
        size_,
        seed_,
        threads_,
        max_retries_
    );
  }

  TrainingSpec::Ptr TrainingSpec::Builder::make() {
    return std::make_shared<TrainingSpec>(build());
  }

  namespace {
    /**
     * @brief Error if @p strategy does not support @p mode.
     *
     * Uses `user_error` because the failure is driven by user-assembled
     * input (CLI flags, config JSON, R strategy wrappers). The mismatch
     * surfaces as a clean error message rather than a stack-traced
     * internal exception.
     */
    template<typename StrategyPtr>
    void check_supports(StrategyPtr const& strategy, std::string const& family, types::Mode mode) {
      if (!strategy) {
        return;
      }

      auto const modes     = strategy->supported_modes();
      auto const supported = modes.find(mode) != modes.end();
      auto const name      = strategy->display_name();

      user_error(supported, family + " strategy '" + name + "' does not support " + types::to_string(mode) + " mode.");
    }
  }

  TrainingSpec::TrainingSpec(
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
  )
      : pp(std::move(pp))
      , vars(std::move(vars))
      , cutpoint(std::move(cutpoint))
      , stop(std::move(stop))
      , binarize(std::move(binarize))
      , grouping(std::move(grouping))
      , leaf(std::move(leaf))
      , mode(mode)
      , size(size)
      , seed(seed)
      , threads(threads)
      , max_retries(max_retries) {
    // Mode-strategy compatibility check. Fails fast at spec construction.
    check_supports(this->pp, "pp", mode);
    check_supports(this->vars, "vars", mode);
    check_supports(this->cutpoint, "cutpoint", mode);
    check_supports(this->stop, "stop", mode);
    check_supports(this->binarize, "binarize", mode);
    check_supports(this->grouping, "grouping", mode);
    check_supports(this->leaf, "leaf", mode);
  }


  nlohmann::json TrainingSpec::to_json() const {
    nlohmann::json j = {
        {"pp", pp->to_json()},
        {"vars", vars->to_json()},
        {"cutpoint", cutpoint->to_json()},
        {"stop", stop->to_json()},
        {"binarize", binarize->to_json()},
        {"grouping", grouping->to_json()},
        {"leaf", leaf->to_json()},
        {"mode", types::to_string(mode)},
        {"size", size},
        {"seed", seed},
        {"threads", threads},
        {"max_retries", max_retries},
    };

    return j;
  }

  TrainingSpec::Ptr TrainingSpec::from_json(nlohmann::json const& j) {
    types::Mode const mode = types::mode_from_string(j.value("mode", "classification"));

    return builder(mode)
        .size(j.value("size", 0))
        .seed(j.value("seed", 0))
        .threads(j.value("threads", 0))
        .max_retries(j.value("max_retries", 3))
        .pp(pp::ProjectionPursuit::from_json(j.at("pp")))
        .vars(vars::VariableSelection::from_json(j.at("vars")))
        .cutpoint(cutpoint::Cutpoint::from_json(j.at("cutpoint")))
        .stop(stop::StopRule::from_json(j.at("stop")))
        .binarize(binarize::Binarization::from_json(j.at("binarize")))
        .grouping(grouping::Grouping::from_json(j.at("grouping")))
        .leaf(leaf::LeafStrategy::from_json(j.at("leaf")))
        .make();
  }

  // -- Strategy forwarders with postcondition checks -----------------------
  // Each forwarder asserts that the strategy upheld its write contract on
  // `NodeContext`. That way UB from a misbehaving strategy is caught at the
  // producer boundary, not at some later consumer's deref.

  bool TrainingSpec::should_stop(NodeContext const& ctx, stats::RNG& rng) const {
    // Tree-level invariant: a single-group node cannot be split by any
    // grouping strategy. Regression's `ByCutpoint` can produce such a
    // child (e.g. when all rows on one side of the cutpoint share the
    // same response), and the regression-default stop rules (`MinSize`,
    // `MinVariance`) won't catch it — short-circuit here so the tree
    // builder doesn't need a special case.
    if (ctx.y.groups.size() < 2) {
      return true;
    }
    return (*stop)(ctx, rng);
  }

  void TrainingSpec::find_projection(NodeContext& ctx, stats::RNG& rng) const {
    (*pp)(ctx, rng);
    if (ctx.aborted) {
      return;
    }
    invariant(ctx.projector.has_value(), "ProjectionPursuit must set ctx.projector");
    invariant(ctx.pp_index_value.has_value(), "ProjectionPursuit must set ctx.pp_index_value");
    if (ctx.projector->hasNaN()) {
      ctx.aborted = true;
    }
  }

  void TrainingSpec::select_vars(NodeContext& ctx, stats::RNG& rng) const {
    (*vars)(ctx, rng);
    if (ctx.aborted) {
      return;
    }
    invariant(ctx.var_selection.has_value(), "VariableSelection must set ctx.var_selection");
  }

  void TrainingSpec::find_cutpoint(NodeContext& ctx, stats::RNG& rng) const {
    (*cutpoint)(ctx, rng);
    if (ctx.aborted) {
      return;
    }
    invariant(ctx.cutpoint.has_value(), "Cutpoint must set ctx.cutpoint");

    // Post-hoc: orient the two active-partition groups by projected mean so
    // `lower_group` routes to the lower child and `upper_group` to the upper.
    // This is a uniform step across all cutpoint strategies — no strategy
    // should worry about it. Grouping consumes ctx.lower_group / upper_group.
    auto const& y_part = ctx.active_partition();
    invariant(y_part.groups.size() == 2, "find_cutpoint expects a 2-group active partition");

    auto it              = y_part.groups.begin();
    types::GroupId lower = *it;
    types::GroupId upper = *std::next(it);

    types::Feature const mean_lower = y_part.group(ctx.x, lower).colwise().mean().dot(*ctx.projector);
    types::Feature const mean_upper = y_part.group(ctx.x, upper).colwise().mean().dot(*ctx.projector);

    if (mean_lower > mean_upper) {
      std::swap(lower, upper);
    }

    ctx.lower_group = lower;
    ctx.upper_group = upper;
  }

  void TrainingSpec::regroup(NodeContext& ctx, stats::RNG& rng) const {
    (*binarize)(ctx, rng);
    if (ctx.aborted) {
      return;
    }
    invariant(ctx.y_bin.has_value(), "Binarization must set ctx.y_bin");
    if (ctx.y_bin->groups.size() < 2) {
      ctx.aborted = true;
    }
  }

  void TrainingSpec::group(NodeContext& ctx, stats::RNG& rng) const {
    if (ctx.aborted) {
      return;
    }

    invariant(
        ctx.lower_group.has_value() && ctx.upper_group.has_value(),
        "group() requires ctx.lower_group and ctx.upper_group (set by find_cutpoint)"
    );
    (*grouping)(ctx, *ctx.lower_group, *ctx.upper_group, rng);

    invariant(
        ctx.lower_y_part.has_value() && ctx.upper_y_part.has_value(),
        "Grouping must set ctx.lower_y_part and ctx.upper_y_part"
    );

    // "No progress" = every row went to one side of the split. Recursing on
    // an identical partition would be unbounded; mark the context aborted
    // so the orchestrator writes a degenerate leaf.
    int const parent_size = ctx.y.total_size();
    if (ctx.lower_y_part->total_size() >= parent_size || ctx.upper_y_part->total_size() >= parent_size) {
      ctx.aborted = true;
    }
  }
}
