#include "utils/Invariant.hpp"

#include <stdexcept>

void invariant(bool condition, char const* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void invariant(bool condition, std::string const& message) {
  invariant(condition, message.c_str());
}
