#include "models/TrainingSpec.hpp"

#include <stdexcept>

namespace ppforest2 {

  TrainingSpec::TrainingSpec(
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
  )
      : pp(std::move(pp))
      , vars(std::move(vars))
      , cutpoint(std::move(cutpoint))
      , stop(std::move(stop))
      , binarize(std::move(binarize))
      , partition(std::move(partition))
      , leaf(std::move(leaf))
      , size(size)
      , seed(seed)
      , threads(threads)
      , max_retries(max_retries) {}

  nlohmann::json TrainingSpec::to_json() const {
    return {
        {"pp", pp->to_json()},
        {"vars", vars->to_json()},
        {"cutpoint", cutpoint->to_json()},
        {"stop", stop->to_json()},
        {"binarize", binarize->to_json()},
        {"partition", partition->to_json()},
        {"leaf", leaf->to_json()},
        {"size", size},
        {"seed", seed},
        {"threads", threads},
        {"max_retries", max_retries},
    };
  }

  TrainingSpec::Ptr TrainingSpec::from_json(nlohmann::json const& j) {
    return builder()
        .size(j.value("size", 0))
        .seed(j.value("seed", 0))
        .threads(j.value("threads", 0))
        .max_retries(j.value("max_retries", 3))
        .pp(pp::ProjectionPursuit::from_json(j.at("pp")))
        .vars(vars::VariableSelection::from_json(j.at("vars")))
        .cutpoint(cutpoint::SplitCutpoint::from_json(j.at("cutpoint")))
        .stop(stop::StopRule::from_json(j.at("stop")))
        .binarize(binarize::Binarization::from_json(j.at("binarize")))
        .partition(partition::StepPartition::from_json(j.at("partition")))
        .leaf(leaf::LeafStrategy::from_json(j.at("leaf")))
        .make();
  }
}
