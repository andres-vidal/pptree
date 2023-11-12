CONAN_DIR=_conan

CORE_DIR=core
CORE_FILES := $(find $CORE_DIR -type f -o -name '*.cpp' -o -name '*.hpp')

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

build: 
	@cmake \
		-DCMAKE_TOOLCHAIN_FILE=../$(CONAN_DIR)/conan_toolchain.cmake \
		-DCMAKE_BUILD_TYPE=Release \
		-S=$(CORE_DIR) \
		-B=$(BUILD_DIR)
	@cd $(BUILD_DIR) && make

clean:
	rm -rf _build

clean-env:
	rm -rf _conan

clean-all: clean clean-env

run:
	@./$(BUILD_DIR)/pptree-cli