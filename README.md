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

### Using asdf version manager

1. Install asdf following [the docs](https://asdf-vm.com/guide/getting-started.html)
2. Install asdf conan plugin following [the docs](https://github.com/amrox/asdf-pyapp#compatible-python-applications)

## Project Automation

Project automation is achieved via Make. Useful scripts are defined in the `Makefile`.

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

Tests can be run using the following command from the project's root:

```bash
make test
```

## Troubleshooting

#### Header files downloaded with Conan not found in VSCode

Add Conan data directory to `includePath` in VSCode. The path is `~/.conan2/**`.
