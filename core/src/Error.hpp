#pragma once

#include <exception>
#include <stdexcept>
#include <string>
#include <sstream>

#include "Logger.hpp"


namespace models {
  class training_error : public std::runtime_error {
    public:
      explicit training_error(const std::string & message) :  std::runtime_error(message) {
      }
  };

  template <typename Catch, typename Throw = Catch, typename Function>
  auto attempt(int max_retries, Function f) -> decltype(f()) {
    int attempts = 0;

    while (attempts < max_retries) {
      try {
        return f();
      } catch (const Catch& e) {
        attempts++;

        if (attempts == max_retries) {
          std::stringstream ss;
          ss << "Failed to execute function after " << max_retries << " attempts." << std::endl;
          ss << "Last exception was: " << e.what();
          throw Throw(ss.str());
        }

        LOG_INFO << "Caught exception: " << e.what() << std::endl;
        LOG_INFO << "Retrying... (" << attempts << "/" << max_retries << ")" << std::endl;
      }
    }

    return f();
  }
}
