add_library(pptree-serialization OBJECT
  Json.cpp)

target_link_libraries(pptree-serialization PRIVATE pptree-compile nlohmann_json::nlohmann_json)
