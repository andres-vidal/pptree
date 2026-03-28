/**
 * @file CLI.cpp
 * @brief Main entry point for the ppforest2 command-line tool.
 */
#include "cli/CLIOptions.hpp"
#include "cli/Train.hpp"
#include "cli/Predict.hpp"
#include "cli/Evaluate.hpp"
#include "cli/Benchmark.hpp"
#include "cli/Summarize.hpp"
#include "io/Color.hpp"
#include "io/IO.hpp"

#include <fmt/format.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ppforest2::cli;

int main(int argc, char *argv[]) {
  CLIOptions params = parse_args(argc, argv);

  ppforest2::io::style::init_color(params.no_color);

  // Post-parse: ensure .json extension on output paths
  if (!params.save_path.empty()) {
    params.save_path = ppforest2::io::json::ensure_extension(params.save_path);
  }

  if (!params.output_path.empty()) {
    params.output_path = ppforest2::io::json::ensure_extension(params.output_path);
  }

  #ifdef _OPENMP
  omp_set_num_threads(params.model.threads);
  #endif

  switch (params.subcommand) {
    case Subcommand::train:     return run_train(params);

    case Subcommand::predict:   return run_predict(params);

    case Subcommand::evaluate:  return run_evaluate(params);

    case Subcommand::benchmark: return run_benchmark(params, argv[0]);

    case Subcommand::summarize: return run_summarize(params);

    default:
      fmt::print(stderr, "Error: No subcommand specified\n");
      return 1;
  }
}
