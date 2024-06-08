# pptree

## Project Structure

```yaml
core # C++ core library
 ├── pptree
 │     ├── include
 │     │   └── pptree # Public header files
 │     │         └── ...
 │     └── src  # Private source files
 │        ├── ...
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

## Environment Setup (Windows)

Install project build dependencies:

- `cmake` (Minimum version: 3.14)
- `make`

Install a `MinGW` with `gcc` and `gfortran` suitable for your architecture. Ensure the `bin` directory of `MinGW` where `gcc.exe` and `gfortran.exe` are located is in the system's PATH. The compiler must be `gcc` in order to build the `R` package. Microsoft Visual Studio can be used to compile the core library, but it will not be linkable to the `R` package.

Install a unix-like terminal like `bash` to interact with the project. With `Rtools` in the path, `make` and the utilities needed to run the `Makefile` will be available from `cmd` and `PowerShell` without having to install `bash` or alike.

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

# R package management

Install project build dependencies:

- `R`
- `Rtools` for your `R` version (Windows Only)
- `TinyTex`, or other `TeX` distribution with `collection-fontsrecommended` and `psnfss` packages`

Ensure `R` and `Rtools` are in the system's PATH, as well as `pdflatex` provided by your `TeX` distribution. 

Install the `R` package's dependencies running executing:

- `make r-install-deps`

The `Makefile` defines useful targets to build, check and install the R package from the project's root:

- `make r-build` prepares the source code and runs `R CMD build`
- `make r-check` runs `R CMD check` on the built tarball
- `make r-install` runs `R CMD build` on the built tarball
- `make r-clean` removes compilation byproducts from the R package
- `make r-document` updates the documentation based on source files

**Do not check or install the package from the raw source files. Always run `make r-build` and use the generated tarball.** This is important, because that target copies the core library's code to the package so it can be compiled on install.


## Runnings Tests

Test can be run using one of the following `Makefile` targets:

- `make test`
- `make test-debug`

