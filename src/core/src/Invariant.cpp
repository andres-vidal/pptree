#include "Invariant.hpp"

#include <stdexcept>

void invariant(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}
