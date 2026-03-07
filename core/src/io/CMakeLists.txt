add_library(pptree-io OBJECT IO.cpp Presentation.cpp)

target_link_libraries(pptree-io PUBLIC
  pptree-compile
  nlohmann_json::nlohmann_json
  csv
  fmt::fmt)

target_include_directories(pptree-io PUBLIC
  ${csv_SOURCE_DIR}/include)
