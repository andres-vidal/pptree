#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

/**
 * @brief CRTP base class providing self-registration for strategy types.
 *
 * Each strategy family (PP, DR, SR) inherits from Strategy<Derived>
 * to get a shared_ptr Ptr typedef, a static registry of named factory
 * functions, and a from_json() dispatcher that looks up the "name"
 * field and delegates to the registered factory.
 *
 * Concrete strategies register themselves at static-init time via
 * the PPFOREST2_REGISTER_STRATEGY macro.
 *
 * @tparam Derived  The strategy base class (e.g. PPStrategy).
 */
template<typename Derived> struct Strategy {
  using Ptr = std::shared_ptr<Derived>;

  /** @brief Factory function type for deserializing a strategy from JSON. */
  using Factory = Ptr (*)(nlohmann::json const&);

  virtual ~Strategy() = default;

  /**
   * @brief Serialize this strategy's configuration to JSON.
   *
   * Must include a "name" field identifying the strategy type,
   * plus any strategy-specific parameters.
   *
   * @param j  JSON object to populate.
   */
  virtual void to_json(nlohmann::json& j) const = 0;

  /** @brief Human-readable name for display in summaries. */
  virtual std::string display_name() const = 0;

  /**
   * @brief Register a concrete strategy for JSON deserialization.
   *
   * Called automatically by concrete strategies via inline static
   * initialization.  Adding a new strategy only requires defining
   * a static `from_json` and the PPFOREST2_REGISTER_STRATEGY macro.
   *
   * @param name     Strategy name (must match the "name" field in JSON).
   * @param factory  Factory function that constructs the strategy.
   * @return         Always true (return value used for static init).
   */
  static bool register_strategy(std::string const& name, Factory factory) {
    registry()[name] = factory;
    return true;
  }

  /**
   * @brief Construct a strategy from its JSON representation.
   *
   * Dispatches on the "name" field to the registered factory.
   *
   * @param j  JSON object (must contain "name").
   * @return   Shared pointer to the constructed strategy.
   */
  static Ptr from_json(nlohmann::json const& j) {
    auto name = j.at("name").get<std::string>();
    auto& reg = registry();
    auto it   = reg.find(name);

    if (it == reg.end())
      throw std::runtime_error("Unknown strategy: " + name);

    return it->second(j);
  }

private:
  static std::map<std::string, Factory>& registry() {
    static std::map<std::string, Factory> r;
    return r;
  }
};

/**
 * @brief Auto-registration macro for strategy factories.
 *
 * Registers `ConcreteStrategy::from_json` as the factory for @p name
 * in the given @p StrategyBase class.
 *
 * Usage (inside the concrete strategy struct, after from_json):
 * @code
 *   struct PPPDAStrategy : public PPStrategy {
 *     static PPStrategy::Ptr from_json(const nlohmann::json& j);
 *     PPFOREST2_REGISTER_STRATEGY(PPStrategy, "pda")
 *   };
 * @endcode
 */
#define PPFOREST2_REGISTER_STRATEGY(StrategyBase, name) \
  inline static const bool registered_ = StrategyBase::register_strategy(name, from_json);
