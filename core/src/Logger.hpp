#pragma once

#include <iostream>

#ifdef NDEBUG
  #define LOG_INFO    if (false) std::cout
  #define LOG_DEBUG   if (false) std::cout
  #define LOG_WARNING if (false) std::cout
#else
  #define LOG_INFO    std::cout << "[INFO]" << "[" << __FUNCTION__ << "] "
  #define LOG_DEBUG   std::cout << "[DEBUG]" << "[" << __FUNCTION__ << "] "
  #define LOG_WARNING std::cout << "[WARNING]" << "[" << __FUNCTION__ << "] "
#endif
