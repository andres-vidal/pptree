add_library(ppforest2-io OBJECT IO.cpp EvaluateResult.cpp Presentation.cpp)

target_link_libraries(ppforest2-io PUBLIC
  ppforest2-compile
  nlohmann_json::nlohmann_json
  csv
  fmt::fmt)

target_include_directories(ppforest2-io PUBLIC
  ${csv_SOURCE_DIR}/include)
