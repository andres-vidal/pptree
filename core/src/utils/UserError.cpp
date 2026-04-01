/**
 * @file UserError.cpp
 * @brief Implementation of user-facing input validation errors.
 */
#include "utils/UserError.hpp"

namespace ppforest2 {
  void user_error(bool condition, std::string const& message) {
    if (!condition) {
      throw UserError(message);
    }
  }

  void user_error(bool condition, char const* message) {
    if (!condition) {
      throw UserError(message);
    }
  }
}
