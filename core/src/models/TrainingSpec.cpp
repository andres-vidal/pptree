#include "models/TrainingSpec.hpp"

#include <stdexcept>

namespace ppforest2 {
  TrainingSpec::TrainingSpec(
      pp::PPStrategy::Ptr pp,
      dr::DRStrategy::Ptr dr,
      sr::SRStrategy::Ptr sr,
      int size,
      int seed,
      int threads,
      int max_retries
  )
      : pp_strategy(std::move(pp))
      , dr_strategy(std::move(dr))
      , sr_strategy(std::move(sr))
      , size(size)
      , seed(seed)
      , threads(threads)
      , max_retries(max_retries) {}

  void TrainingSpec::to_json(nlohmann::json& j) const {
    nlohmann::json pp_json, dr_json, sr_json;
    pp_strategy->to_json(pp_json);
    dr_strategy->to_json(dr_json);
    sr_strategy->to_json(sr_json);

    j["pp"]          = pp_json;
    j["dr"]          = dr_json;
    j["sr"]          = sr_json;
    j["size"]        = size;
    j["seed"]        = seed;
    j["threads"]     = threads;
    j["max_retries"] = max_retries;
  }

  TrainingSpec::Ptr TrainingSpec::from_json(nlohmann::json const& j) {
    return make(
        pp::PPStrategy::from_json(j.at("pp")),
        dr::DRStrategy::from_json(j.at("dr")),
        sr::SRStrategy::from_json(j.at("sr")),
        j.value("size", 0),
        j.value("seed", 0),
        j.value("threads", 0),
        j.value("max_retries", 3)
    );
  }
}
