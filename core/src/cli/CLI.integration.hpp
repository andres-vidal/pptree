/**
 * @file CLI.integration.hpp
 * @brief Shared utilities for CLI integration tests.
 *
 * Each test spawns the ppforest2 binary as a child process and checks its
 * exit code and stdout output.  Temporary files (TempFile, TempDir) are
 * used for model and output artifacts; they are automatically cleaned up.
 */
#pragma once

#include <gtest/gtest.h>

#include "io/TempFile.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>
#include <filesystem>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

#ifndef PPFOREST2_BINARY_PATH
#error "PPFOREST2_BINARY_PATH must be defined"
#endif

#ifndef PPFOREST2_DATA_DIR
#error "PPFOREST2_DATA_DIR must be defined"
#endif

#ifndef PPFOREST2_GOLDEN_DIR
#error "PPFOREST2_GOLDEN_DIR must be defined"
#endif

inline const std::string BINARY     = PPFOREST2_BINARY_PATH;
inline const std::string DATA_DIR   = PPFOREST2_DATA_DIR;
inline const std::string GOLDEN_DIR = PPFOREST2_GOLDEN_DIR;
inline const std::string IRIS_CSV   = DATA_DIR + "/iris.csv";
inline const std::string CRAB_CSV   = DATA_DIR + "/crab.csv";
inline const std::string WINE_CSV   = DATA_DIR + "/wine.csv";
inline const std::string GLASS_CSV  = DATA_DIR + "/glass.csv";

/**
 * @brief Captured output of a child-process invocation.
 *
 * Holds the exit code and the entire stdout text so tests can assert
 * both process success and textual output content.
 */
struct ProcessResult {
  int exit_code;
  std::string stdout_output;
};

/**
 * @brief Spawn the ppforest2 binary with the given argument string.
 *
 * Stderr is redirected to /dev/null (NUL on Windows) so only stdout
 * is captured.  The exit code is extracted via WEXITSTATUS on POSIX.
 *
 * @param args  Space-separated argument string appended to the binary path.
 * @return ProcessResult with exit code and captured stdout.
 */
inline ProcessResult run_ppforest2(const std::string& args) {
  #ifdef _WIN32
  std::string cmd = BINARY + " " + args + " 2>NUL";
  FILE *pipe      = _popen(cmd.c_str(), "r");
  #else
  std::string cmd = BINARY + " " + args + " 2>/dev/null";
  FILE *pipe      = popen(cmd.c_str(), "r");
  #endif

  if (!pipe) {
    return { -1, "" };
  }

  std::string output;
  char buffer[4096];

  while (fgets(buffer, sizeof(buffer), pipe))
    output += buffer;

  #ifdef _WIN32
  int exit_code = _pclose(pipe);
  #else
  int status    = pclose(pipe);
  int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  #endif

  return { exit_code, output };
}

using ppforest2::io::TempFile;
using ppforest2::io::TempDir;

/**
 * @brief Test fixture that trains a forest model once in SetUp().
 *
 * Provides the model file path and parsed JSON for structural assertions
 * without repeating the training step in each test.
 */
class SavedModelTest : public ::testing::Test {
  protected:
    void SetUp() override {
      model_.reset(new TempFile());
      model_->clear();
      auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -s " + model_->path());
      ASSERT_EQ(result.exit_code, 0);
      model_json_ = json::parse(model_->read());
    }

    std::unique_ptr<TempFile> model_;
    json model_json_;
};
