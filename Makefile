OUTPUT_DIR=_conan

CORE_FILES := $(find core -type f -o -name '*.cpp' -o -name '*.hpp')

install:
	@conan install . --output-folder=$(OUTPUT_DIR) --build=missing && pre-commit install

lint:
ifeq (,$(CORE_FILES))
	@echo "No files to lint"
else
	@cppcheck ./core/**/*.cpp ./core/**/*.hpp
endif

format:
ifeq (,$(CORE_FILES))
	@echo "No files to format"
else
	@uncrustify ./core/**/*.cpp ./core/**/*.hpp -c uncrustify.cfg --no-backup --replace
endif
