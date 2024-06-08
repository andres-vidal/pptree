cmake_minimum_required(VERSION 3.14)
project(DevelopmentTools)

include(ExternalProject)

# Setup Cppcheck
ExternalProject_Add(
    cppcheck
    GIT_REPOSITORY https://github.com/danmar/cppcheck.git
    GIT_TAG 2.14.0
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}/cppcheck
)

# Setup Uncrustify
ExternalProject_Add(
    uncrustify
    GIT_REPOSITORY https://github.com/uncrustify/uncrustify.git
    GIT_TAG uncrustify-0.78.1
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}/uncrustify
)

# Example usage after build
add_custom_target(analyze
    COMMAND ${CMAKE_BINARY_DIR}/cppcheck/bin/cppcheck --enable=all --check-level=exhaustive --suppress=missingIncludeSystem --quiet ${CMAKE_SOURCE_DIR}/../core -I${CMAKE_SOURCE_DIR}/../core/src -I${CMAKE_SOURCE_DIR}/../core/include
    DEPENDS cppcheck
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(format
    COMMAND ${CMAKE_BINARY_DIR}/uncrustify/bin/uncrustify -c ${CMAKE_SOURCE_DIR}/../uncrustify.cfg  ${CMAKE_SOURCE_DIR}/../core/**/*.cpp  ${CMAKE_SOURCE_DIR}/../core/**/*.hpp --no-backup
    DEPENDS uncrustify
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(format-dry
    COMMAND ${CMAKE_BINARY_DIR}/uncrustify/bin/uncrustify -c ${CMAKE_SOURCE_DIR}/../uncrustify.cfg ${CMAKE_SOURCE_DIR}/../core/**/*.cpp  ${CMAKE_SOURCE_DIR}/../core/**/*.hpp --check
    DEPENDS uncrustify
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
