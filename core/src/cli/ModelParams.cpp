/**
 * @file ModelParams.cpp
 * @brief Model parameter construction, resolution, and defaults.
 */
#include "cli/ModelParams.hpp"
#include "cli/JsonApply.hpp"
#include "cli/Validation.hpp"
#include "utils/UserError.hpp"

#include <charconv>
#include <cmath>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <regex>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ppforest2::cli {

  namespace {
    void apply_strategy(
        nlohmann::json const& obj, std::string const& key, nlohmann::json& target_config, std::string& target_input
    ) {
      if (!obj.contains(key)) {
        return;
      }

      if (obj[key].is_object()) {
        target_config = obj[key];
      } else if (obj[key].is_string()) {
        target_input = obj[key].get<std::string>();
      }
    }

    nlohmann::json parse_scalar(std::string const& s) {
      int ival;
      auto [iptr, iec] = std::from_chars(s.data(), s.data() + s.size(), ival);

      if (iec == std::errc{} && iptr == s.data() + s.size()) {
        return ival;
      }

      char* end   = nullptr;
      double dval = std::strtod(s.c_str(), &end);

      if (end == s.c_str() + s.size()) {
        return dval;
      }

      return s;
    }

    void try_parse_strategy(std::string const& input, std::string const& flag, nlohmann::json& config) {
      if (input.empty()) {
        return;
      }

      try {
        config = strategy_string_to_json(input);
      } catch (std::exception const& e) {
        throw ppforest2::UserError(fmt::format("Invalid --{} value: {}", flag, e.what()));
      }
    }

    int proportion_to_count(float p, unsigned int total) {
      return static_cast<int>(std::round(static_cast<float>(total) * p));
    }
  }

  float parse_proportion(std::string const& input) {
    try {
      auto slash_pos = input.find('/');

      if (slash_pos != std::string::npos) {
        int numerator   = std::stoi(input.substr(0, slash_pos));
        int denominator = std::stoi(input.substr(slash_pos + 1));

        user_error(denominator > 0, "fraction denominator must be positive");
        user_error(numerator > 0, "fraction numerator must be positive");

        float val = static_cast<float>(numerator) / static_cast<float>(denominator);

        user_error(val <= 1, "fraction must evaluate to a proportion in (0, 1]");

        return val;
      }

      float val = std::stof(input);

      user_error(val > 0 && val <= 1, "proportion must be in (0, 1]");

      return val;
    } catch (ppforest2::UserError const&) {
      throw;
    } catch (std::exception const& e) {
      throw ppforest2::UserError(fmt::format("Invalid proportion '{}': {}", input, e.what()));
    }
  }

  float parse_proportion(nlohmann::json const& j) {
    if (j.is_string()) {
      return parse_proportion(j.get<std::string>());
    }

    float val = j.get<float>();

    user_error(val > 0 && val <= 1, "proportion must be in (0, 1]");

    return val;
  }

  nlohmann::json strategy_string_to_json(std::string const& input) {
    static std::regex const pattern(R"(([^:]+)(?::(.+))?)");
    static std::regex const param_pattern(R"(([^=,]+)=([^,]*))");

    std::smatch match;

    if (!std::regex_match(input, match, pattern)) {
      throw std::runtime_error("Invalid strategy string: " + input);
    }

    nlohmann::json j;
    j["name"] = match[1].str();

    if (!match[2].matched) {
      return j;
    }

    std::string params_str = match[2].str();

    // Validate that all comma-separated tokens are key=value pairs
    static std::regex const token_pattern(R"([^,]+)");
    static std::regex const kv_pattern(R"(([^=]+)=(.*))");

    auto begin = std::sregex_iterator(params_str.begin(), params_str.end(), token_pattern);
    auto end   = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
      std::string token = (*it).str();
      std::smatch kv;

      if (!std::regex_match(token, kv, kv_pattern)) {
        throw std::runtime_error("Invalid parameter (expected key=value): " + token);
      }

      j[kv[1].str()] = parse_scalar(kv[2].str());
    }

    return j;
  }

  void ModelParams::validate(nlohmann::json const& config, std::vector<std::string>& errors) {
    // Size
    if (config.contains("size")) {
      check(config["size"].is_number_integer(), "size must be an integer", errors);

      if (config["size"].is_number_integer()) {
        check(config["size"].get<int>() >= 0, "size must be >= 0", errors);
      }
    } else {
      errors.push_back("size is required");
    }

    // Lambda
    if (config.contains("lambda") && config["lambda"].is_number()) {
      auto lambda = config["lambda"].get<float>();
      check(lambda >= 0 && lambda <= 1, "lambda must be in [0, 1]", errors);
    }

    // PP strategy vs lambda: at least one must be present
    check(config.contains("pp") || config.contains("lambda"), "lambda is required (or provide a pp strategy)", errors);

    // Variable selection
    if (config.contains("p_vars") && config["p_vars"].is_number()) {
      auto pv = config["p_vars"].get<float>();
      check(pv > 0 && pv <= 1, "p_vars must be in (0, 1]", errors);
    }

    bool has_vars_strategy = config.contains("vars") && !config["vars"].is_null();

    if (config.contains("n_vars") && config["n_vars"].is_number_integer() && !has_vars_strategy) {
      check(config["n_vars"].get<int>() > 0, "n_vars must be positive", errors);
    }

    // Threads
    if (config.contains("threads") && config["threads"].is_number_integer()) {
      check(config["threads"].get<int>() > 0, "threads must be positive", errors);
    }

    // Max retries
    if (config.contains("max_retries") && config["max_retries"].is_number_integer()) {
      check(config["max_retries"].get<int>() >= 0, "max_retries must be non-negative", errors);
    }
  }

  ModelParams::ModelParams(nlohmann::json const& config) {
    apply(config, "size", size);
    apply(config, "lambda", lambda);
    apply(config, "seed", seed);
    apply(config, "threads", threads);
    apply(config, "max_retries", max_retries);
    apply(config, "n_vars", n_vars);

    if (config.contains("p_vars")) {
      auto const& v = config["p_vars"];
      p_vars_input  = v.is_string() ? v.get<std::string>() : v.dump();
    }

    apply_strategy(config, "pp", pp_config, pp_input);
    apply_strategy(config, "vars", vars_config, vars_input);
    apply_strategy(config, "cutpoint", cutpoint_config, cutpoint_input);
    apply_strategy(config, "stop", stop_config, stop_input);
    apply_strategy(config, "binarize", binarize_config, binarize_input);
    apply_strategy(config, "partition", partition_config, partition_input);
    apply_strategy(config, "leaf", leaf_config, leaf_input);
  }

  void ModelParams::resolve() {
    try_parse_strategy(pp_input, "pp", pp_config);
    try_parse_strategy(vars_input, "vars", vars_config);
    try_parse_strategy(cutpoint_input, "cutpoint", cutpoint_config);
    try_parse_strategy(stop_input, "stop", stop_config);
    try_parse_strategy(binarize_input, "binarize", binarize_config);
    try_parse_strategy(partition_input, "partition", partition_config);
    try_parse_strategy(leaf_input, "leaf", leaf_config);

    if (!p_vars_input.empty()) {
      p_vars = parse_proportion(p_vars_input);
    }
  }

  void ModelParams::resolve_defaults(unsigned int total_vars) {
    // clang-format off
    if (!threads) {
      #ifdef _OPENMP
      threads = omp_get_max_threads();
      #else
      threads = 1;
      #endif
    }
    // clang-format on

    if (total_vars == 0) {
      return;
    }

    // Resolve implicit APIs into explicit strategy configs
    if (pp_config.is_null()) {
      pp_config = {{"name", "pda"}, {"lambda", lambda}};
    }

    if (vars_config.is_null() && size > 0) {
      if (n_vars) {
        p_vars      = static_cast<float>(*n_vars) / static_cast<float>(total_vars);
        vars_config = {{"name", "uniform"}, {"count", *n_vars}};
      } else if (p_vars) {
        n_vars      = proportion_to_count(*p_vars, total_vars);
        vars_config = {{"name", "uniform"}, {"count", *n_vars}};
      } else {
        p_vars      = 0.5F;
        n_vars      = proportion_to_count(*p_vars, total_vars);
        vars_config = {{"name", "uniform"}, {"count", *n_vars}};
      }
    }

    // Resolve proportion to count in explicit vars config
    if (!vars_config.is_null() && vars_config.contains("proportion")) {
      int count = proportion_to_count(vars_config["proportion"].get<float>(), total_vars);
      vars_config.erase("proportion");
      vars_config["count"] = count;
    }

    // Strategy config defaults (pp and vars are resolved above via implicit APIs)
    nlohmann::json defaults = {
        {"pp", {{"name", "pda"}}},
        {"vars", {{"name", "all"}}},
        {"cutpoint", {{"name", "mean_of_means"}}},
        {"stop", {{"name", "pure_node"}}},
        {"binarize", {{"name", "largest_gap"}}},
        {"partition", {{"name", "by_group"}}},
        {"leaf", {{"name", "majority_vote"}}},
    };

    auto fill = [&](std::string const& key, nlohmann::json& config) {
      config = config.is_null() ? defaults[key] : config;
    };

    fill("pp", pp_config);
    fill("vars", vars_config);
    fill("cutpoint", cutpoint_config);
    fill("stop", stop_config);
    fill("binarize", binarize_config);
    fill("partition", partition_config);
    fill("leaf", leaf_config);
  }
}
