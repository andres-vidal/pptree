#include "Random.hpp"

#include <thread>

std::vector<std::mt19937> models::stats::Random::rngs(std::thread::hardware_concurrency());
