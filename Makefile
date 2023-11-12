OUTPUT_DIR=_conan

install:
	@conan install . --output-folder=$(OUTPUT_DIR) --build=missing

lint:
	@cppcheck ./core/**/*.cpp ./core/**/*.hpp

format:
	@uncrustify ./core/**/*.cpp ./core/**/*.hpp -c uncrustify.cfg --no-backup --replace 

