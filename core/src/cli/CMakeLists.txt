add_library(ppforest2-cli OBJECT
  CLIOptions.cpp
  VarsSpec.cpp
  Train.cpp
  Predict.cpp
  Evaluate.cpp
  Benchmark.cpp
  BenchmarkReport.cpp
  Summarize.cpp
  Metrics.cpp)

target_include_directories(ppforest2-cli PRIVATE
  ${csv_SOURCE_DIR}/include)

target_link_libraries(ppforest2-cli PUBLIC
  ppforest2-compile
  nlohmann_json::nlohmann_json
  CLI11::CLI11
  fmt::fmt)

target_compile_definitions(ppforest2-cli PRIVATE
  PPFOREST2_VERSION="${PROJECT_VERSION}")
