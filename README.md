# pptree

## Project Structure

```yaml
core # C++ core library
 ├── pptree
 │     ├── include
 │     │   └── pptree # Public header files
 │     │         └── ...
 │     ├── src  # Private source files
 │     │   ├── ...
 │     │   ├── ...
 │     │   └── ...
 │     └── test # Test files
 │        ├── ...
 │        └── ...
bindings # Interfaces to other languages
tools # development tools
```

## Environment Setup (Unix)

Install project build dependencies:

- `cmake` (Minimum version: 3.14)
- `make`

Install a `C/C++` compiler if lacking. In order to correctly run the `R` package, the compiler must be the one that `R` uses in your platform. Usually `gcc` in Linux and `clang` in Mac.

Install `R` and make sure its binaries are in the system's PATH.

Install the `R` package's dependencies running executing:

- `make r-install-deps`

## Environment Setup (Windows)

Install project build dependencies:

- `cmake` (Minimum version: 3.14)
- `Rtools` for your `R` version

Install a `MinGW` with `gcc` and `gfortran` suitable for your architecture. Ensure the `bin` directory of `MinGW` where `gcc.exe` and `gfortran.exe` are located is in the system's PATH.

Install a unix-like terminal like `bash` to interact with the project. With `Rtools` in the path, `make` and the utilities needed to run the `Makefile` will be available from `cmd` and `PowerShell` without having to install `bash` or alike.

Install `R` and make sure its binaries are in the system's PATH.

Install the `R` package's dependencies running executing:

- `make r-install-deps`

## Dependency Management

Dependencies are downloaded automatically during the build process using CMake's `FetchContent`. These are defined in `core/CMakeLists.txt`. To build the project, and thus download dependencies, run one of the following `Makefile` targets:

- `make build`
- `make build-debug`

## Development Tools

The project comes with development tools configured via the `tools/CMakeLists.txt` file. Namely, `uncrustify` is used for code formatting and `cppcheck` for static analysis. These can be installed running the `make install-tools` command from the project's root, which will make their binaries available in the `.tools` folder. Configure your IDE to use the binaries in this folder, instead of globally installed ones. The tools can be run via command line using `Makefile` targets:

- `make format`
- `make format-dry`
- `make analyze`

Install the development tools from the project's root executing:

- `make install-tools`

Refer to the root's `Makefile` for other useful commands.

## Runnings Tests

Test can be run using one of the following `Makefile` targets:

- `make test`
- `make test-debug`

