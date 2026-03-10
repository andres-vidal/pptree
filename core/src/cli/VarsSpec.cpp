/**
 * @file VarsSpec.cpp
 * @brief Shared parsing for the --vars / vars parameter.
 */
#include "cli/VarsSpec.hpp"
#include "utils/Invariant.hpp"

namespace pptree::cli {
  VarsSpec parse_vars(const std::string& input) {
    auto slash_pos = input.find('/');

    if (slash_pos != std::string::npos) {
      int numerator   = std::stoi(input.substr(0, slash_pos));
      int denominator = std::stoi(input.substr(slash_pos + 1));

      invariant(denominator > 0, "vars fraction denominator must be positive");
      invariant(numerator > 0, "vars fraction numerator must be positive");

      float val = static_cast<float>(numerator) / static_cast<float>(denominator);

      invariant(val <= 1, "vars fraction must evaluate to a proportion in (0, 1]");

      return { true, val };
    }

    if (input.find('.') != std::string::npos) {
      float val = std::stof(input);

      invariant(val > 0 && val <= 1, "vars proportion must be in (0, 1]");

      return { true, val };
    }

    int val = std::stoi(input);

    invariant(val > 0, "vars count must be positive");

    return { false, static_cast<float>(val) };
  }

  VarsSpec parse_vars(const nlohmann::json& j) {
    if (j.is_string()) {
      return parse_vars(j.get<std::string>());
    }

    if (j.is_number_integer()) {
      int val = j.get<int>();

      invariant(val > 0, "vars count must be positive");

      return { false, static_cast<float>(val) };
    }

    float val = j.get<float>();

    invariant(val > 0 && val <= 1, "vars proportion must be in (0, 1]");

    return { true, val };
  }
}
