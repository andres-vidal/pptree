add_library(pptree-cli OBJECT
  CLIOptions.cpp
  VarsSpec.cpp
  Train.cpp
  Predict.cpp
  Evaluate.cpp
  Benchmark.cpp
  BenchmarkReport.cpp)

target_include_directories(pptree-cli PRIVATE
  ${csv_SOURCE_DIR}/include)

target_link_libraries(pptree-cli PUBLIC
  pptree-compile
  nlohmann_json::nlohmann_json
  CLI11::CLI11
  fmt::fmt)

target_compile_definitions(pptree-cli PRIVATE
  PPTREE_VERSION="${PROJECT_VERSION}")
