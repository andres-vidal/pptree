#pragma once
#include <string>
#include <sstream>

void invariant(bool condition, const char *message);
void invariant(bool condition, const std::string &message);
