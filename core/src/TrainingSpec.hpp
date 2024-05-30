#pragma once

#include "PPStrategy.hpp"
#include "DRStrategy.hpp"

#include <memory>
#include <map>

namespace models {
  struct ITrainingParam {
    virtual ~ITrainingParam() = default;
    virtual std::unique_ptr<ITrainingParam> clone() const = 0;
  };

  template<typename T>
  struct TrainingParam : public ITrainingParam {
    T value;

    explicit TrainingParam(const T value) : value(value) {
    }

    std::unique_ptr<ITrainingParam> clone() const override {
      return std::make_unique<TrainingParam<T> >(value);
    }
  };

  template<typename T>
  struct TrainingParamPointer : public ITrainingParam {
    std::shared_ptr<T> ptr;

    explicit TrainingParamPointer(std::shared_ptr<T> ptr) : ptr(ptr) {
    }

    std::unique_ptr<ITrainingParam> clone() const override {
      return ptr;
    }
  };


  struct TrainingParams {
    std::map<std::string, std::unique_ptr<ITrainingParam> > map;

    TrainingParams() {
    }

    TrainingParams(const TrainingParams& other) {
      for (const auto& [key, value] : other.map) {
        map[key] = value->clone();
      }
    }

    template<typename T>
    void set(const std::string &name, T param) {
      map[name] = std::make_unique<TrainingParam<T> >(param);
    }

    template<typename T>
    void set_ptr(const std::string &name, std::shared_ptr<T> param_ptr) {
      map[name] = std::make_unique<TrainingParamPointer<T> >(param_ptr);
    }

    template<typename T>
    T at(const std::string &name) const {
      if (map.find(name) == map.end()) {
        throw std::out_of_range("Parameter " + name + " not found");
      }

      auto ptr = dynamic_cast<const TrainingParam<T> *>(map.at(name).get());

      if (ptr == nullptr) {
        throw std::out_of_range("Parameter '" + name + "' is not of expected type");
      }

      return ptr->value;
    }

    template<typename T>
    T & from_ptr_at(const std::string& name) const {
      if (map.find(name) == map.end()) {
        throw std::out_of_range("Parameter " + name + " not found");
      }

      auto ptr = dynamic_cast<TrainingParamPointer<T> *>(map.at(name).get());

      if (ptr == nullptr) {
        throw std::out_of_range("Parameter '" + name + "' is not of expected type");
      }

      return *ptr->ptr;
    }
  };

  template<typename T, typename R>
  struct TrainingSpec {
    pp::strategy::PPStrategy<T, R> pp_strategy;
    dr::strategy::DRStrategy<T> dr_strategy;
    std::unique_ptr<TrainingParams> params;

    TrainingSpec(
      const pp::strategy::PPStrategy<T, R> pp_strategy,
      const dr::strategy::DRStrategy<T>    dr_strategy)
      : pp_strategy(pp_strategy),
      dr_strategy(dr_strategy),
      params(std::make_unique<TrainingParams>()) {
    }

    TrainingSpec(const TrainingSpec& other)
      : pp_strategy(other.pp_strategy),
      dr_strategy(other.dr_strategy),
      params(new TrainingParams(*other.params)) {
    }

    static TrainingSpec<T, R> glda(const double lambda) {
      auto training_spec = TrainingSpec<T, R>(pp::strategy::glda<T, R>(lambda), dr::strategy::all<T>());
      training_spec.params->set("lambda", lambda);
      return training_spec;
    }

    static TrainingSpec<T, R> lda() {
      return TrainingSpec<T, R>::glda(0.0);
    }

    static TrainingSpec<T, R> uniform_glda(const int n_vars, const double lambda) {
      auto training_spec = TrainingSpec<T, R>(pp::strategy::glda<T, R>(lambda), dr::strategy::uniform<T>(n_vars));
      training_spec.params->set("n_vars", n_vars);
      training_spec.params->set("lambda", lambda);
      return training_spec;
    }
  };
}
