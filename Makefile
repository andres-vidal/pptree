MAKEFLAGS += --no-print-directory

CONAN_DIR=_conan

CORE_DIR=core
CORE_FILES=$(shell find $(CORE_DIR) -type f -name '*.cpp' -o -name '*.hpp')
BUILD_DIR=_build

install:
	@conan install . --output-folder=$(CONAN_DIR) --build=missing && pre-commit install

lint:
ifeq (,$(CORE_FILES))
	@echo "No files to lint"
else
	@cppcheck $(CORE_DIR)/**/*.cpp ./core/**/*.hpp
endif

format:
ifeq (,$(CORE_FILES))
	@echo "No files to format"
else
	@uncrustify $(CORE_DIR)/**/*.cpp ./core/**/*.hpp -c uncrustify.cfg --no-backup --replace
endif

format-dry:
ifeq (,$(CORE_FILES))
	@echo "No files to format"
else
	@uncrustify $(CORE_DIR)/**/*.cpp ./core/**/*.hpp -c uncrustify.cfg --check
endif

build:
	@cd $(BUILD_DIR) && make

build-cmake: 
	@cmake \
		-DCMAKE_TOOLCHAIN_FILE=../$(CONAN_DIR)/conan_toolchain.cmake \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-S=$(CORE_DIR) \
		-B=$(BUILD_DIR)

build-all: build-cmake build

install-debug:
	@conan install . --output-folder=$(CONAN_DIR) --build=missing --settings=build_type=Debug && pre-commit install

build-cmake-debug:
	@cmake \
		-DCMAKE_TOOLCHAIN_FILE=../$(CONAN_DIR)/conan_toolchain.cmake \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-S=$(CORE_DIR) \
		-B=$(BUILD_DIR)

build-all-debug: build-cmake-debug build

clean:
	rm -rf _build

clean-env:
	rm -rf _conan

clean-all: clean clean-env

run:
	@./$(BUILD_DIR)/pptree-cli

test: build-all-debug
	@cd ./$(BUILD_DIR) && ctest --output-on-failure


test-debug: build-all-debug
	@cd ./$(BUILD_DIR) && lldb pptree-test