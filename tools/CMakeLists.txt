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

# Setup Bison (required by Doxygen, macOS ships an outdated version)
ExternalProject_Add(
    bison
    URL https://ftp.gnu.org/gnu/bison/bison-3.8.2.tar.gz
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=${CMAKE_BINARY_DIR}/bison
    BUILD_COMMAND make
    INSTALL_COMMAND make install
)

# Setup Doxygen
ExternalProject_Add(
    doxygen
    GIT_REPOSITORY https://github.com/doxygen/doxygen.git
    GIT_TAG Release_1_13_2
    DEPENDS bison
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}/doxygen
        -DBISON_EXECUTABLE=${CMAKE_BINARY_DIR}/bison/bin/bison
)

# Example usage after build
add_custom_target(analyze
    COMMAND ${CMAKE_BINARY_DIR}/cppcheck/bin/cppcheck --enable=all --check-level=exhaustive --suppress=missingIncludeSystem --quiet ${CMAKE_SOURCE_DIR}/../core -I${CMAKE_SOURCE_DIR}/../core/src -I${CMAKE_SOURCE_DIR}/../core/include
    DEPENDS cppcheck
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

file(GLOB_RECURSE FORMAT_SOURCES
    "${CMAKE_SOURCE_DIR}/../core/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/../core/src/*.hpp"
    "${CMAKE_SOURCE_DIR}/../core/include/*.hpp"
)

add_custom_target(format
    COMMAND ${CMAKE_BINARY_DIR}/uncrustify/bin/uncrustify -c ${CMAKE_SOURCE_DIR}/../uncrustify.cfg ${FORMAT_SOURCES} --no-backup
    DEPENDS uncrustify
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(format-dry
    COMMAND ${CMAKE_BINARY_DIR}/uncrustify/bin/uncrustify -c ${CMAKE_SOURCE_DIR}/../uncrustify.cfg ${FORMAT_SOURCES} --check
    DEPENDS uncrustify
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
