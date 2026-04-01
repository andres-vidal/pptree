cmake_minimum_required(VERSION 3.14)
project(DevelopmentTools)

include(ExternalProject)

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
