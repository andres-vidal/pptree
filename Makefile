OUTPUT_DIR=_conan

install:
	@conan install . --output-folder=$(OUTPUT_DIR) --build=missing

lint:
	@cppcheck ./core/**/*.cpp ./core/**/*.hpp

