/**
 * @file UserError.hpp
 * @brief Exception type for user-facing input validation errors.
 *
 * Use user_error() when the error is caused by invalid user input
 * (bad CSV data, wrong parameters, missing files). These produce
 * actionable error messages rather than assertion-style crashes.
 *
 * Use invariant() for programmer bugs (conditions that should
 * never happen if the code is correct).
 */
#pragma once

#include <stdexcept>
#include <string>

namespace ppforest2 {
  /**
   * @brief Exception for user-facing input validation errors.
   *
   * Distinct from std::runtime_error so callers can catch user
   * errors separately and provide actionable messages (e.g.,
   * "CSV file has inconsistent columns — expected 5, got 3").
   */
  class UserError : public std::runtime_error {
    public:
      using std::runtime_error::runtime_error;
  };

  /**
   * @brief Throw a UserError if the condition is false.
   *
   * @param condition Condition that must hold for valid input.
   * @param message   Actionable error message for the user.
   */
  void user_error(bool condition, const std::string& message);

  /** @copydoc user_error(bool, const std::string&) */
  void user_error(bool condition, const char *message);
}
