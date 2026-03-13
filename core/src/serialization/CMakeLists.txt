add_library(ppforest2-serialization OBJECT
  Json.cpp)

target_link_libraries(ppforest2-serialization PRIVATE ppforest2-compile nlohmann_json::nlohmann_json)
