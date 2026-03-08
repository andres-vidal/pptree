#pragma once
#include <string>
#include <sstream>

/**
 * @brief Runtime assertion that throws on failure.
 *
 * If @p condition is false, throws a std::runtime_error with
 * the given @p message.  Unlike assert(), these checks remain
 * active in release builds.
 *
 * @param condition  Condition that must hold.
 * @param message    Error message if the condition fails.
 */
void invariant(bool condition, const char *message);

/** @copydoc invariant(bool, const char*) */
void invariant(bool condition, const std::string &message);
