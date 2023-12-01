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
```

## Environment Setup

Install the environment dependencies described in the `.tool-versions` file. The [asdf version manager](https://asdf-vm.com/) is recommended to do this if using UNIX based operative system (Linux or Mac). If using Windows, the [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) and following UNIX-like instructions is recommended for development.

Install `asdf`:

```
brew install asdf
```

Install `cmake`:

```
brew install cmake
```

### Using asdf version manager

1. Install asdf following [the docs](https://asdf-vm.com/guide/getting-started.html)
2. Install asdf conan plugin following [the docs](https://github.com/amrox/asdf-pyapp#compatible-python-applications)

## Project Automation

Project automation is achieved via Make. Useful scripts are defined in the `Makefile`. The most important ones are:

| Command            | Description                                                                                                                                                                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `make install`     | Install dependencies. Tools installed this way won't be available in the command line until the Conan environment is activated. See [Dependency Management](#dependency-management). Also installs [Pre-commit Hooks](#pre-commit-hooks)   |
| `make build-cmake` | Runs the meta-build system (CMake) to generate build specifications and scripts. Is the first step to build the app. It's also necessary for the IDE to resolve include paths correctly. **Must be run when a CMakeLists.txt is changed.** |
| `make build`       | Builds the library, effectively generating the relevant executables.                                                                                                                                                                       |
| `make build-all`   | Runs `make build-cmake` and `make build` in sequence.                                                                                                                                                                                      |
| `make clean`       | Removes the build files inside the `_build` directory. To be regenerated with `make build-all`                                                                                                                                             |
| `make clean-env`   | Removes the conan files inside the `_conan` directory. To be regenerated with `make install`. Does not clear the environment from the termina.                                                                                             |
| `make clean-all`   | Runs `make clean` and `make clean-env` in sequence.                                                                                                                                                                                        |
| `make run`         | Runs the app's entry point, defined in `main.cpp`                                                                                                                                                                                          |
| `make test`        | Builds the app (`make build-all`) and runs all tests using CTest.                                                                                                                                                                          |
| `make format`      | Runs the formatter over every `.cpp` and `.hpp` file, making changes in files with format inconsistent with the definitions in `uncrustify.cfg`.                                                                                           |
| `make lint`        | Runs the code analysis tool over every `.cpp` and `.hpp` file.                                                                                                                                                                             |

## Pre-commit hooks

This project has pre-commit hooks configured to run the formatter, the code analysis tool and the test when a new commit is added. Please ensure each commit leaves the code in a valid state. These hooks are installed during the execution of `make install`.

## Dependency Management

Project dependencies are managed by `conan`. In order to use it, a default profile must be set before the first run, by executing the following command:

```bash
conan profile detect --force
```

The project's dependencies are listed in the `conanfile.txt` and must be installed by running the following command:

```bash
make install
```

In order to use the tools defined in the `tool_requires` section of the `conanfile.txt`, it's necessary to activate the Conan environment in the current terminal:

```bash
source _conan/conanbuild.sh
```

## Running Tests

This project has tests configured with GoogleTest. They can be run using the following command from the project's root:

```bash
make test
```

## Troubleshooting

#### Header files downloaded with Conan not found in VSCode

Add Conan data directory to `includePath` in VSCode. The path is `~/.conan2/**`.
