#include "models/TrainingSpec.hpp"

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

  TrainingSpec::Ptr TrainingSpec::Builder::make() { return std::make_shared<TrainingSpec>(build()); }

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
}
