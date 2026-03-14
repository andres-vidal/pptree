/**
 * @file UserError.cpp
 * @brief Implementation of user-facing input validation errors.
 */
#include "utils/UserError.hpp"

namespace ppforest2 {
  void user_error(bool condition, const std::string& message) {
    if (!condition) {
      throw UserError(message);
    }
  }

  void user_error(bool condition, const char *message) {
    if (!condition) {
      throw UserError(message);
    }
  }
}
