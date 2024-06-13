MAKEFLAGS += --no-print-directory

BUILD_DIR = .build
BUILD_DIR_DEBUG = .debug

NLHOMANN_JSON_HEADERS_PATH = ${BUILD_DIR}/_deps/json-src/include

clean:
	@rm -rf ${BUILD_DIR} ${BUILD_DIR_DEBUG}

build:
	@mkdir -p ${BUILD_DIR}
	@cd ${BUILD_DIR} && cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ../core && make

build-debug:
	@mkdir -p ${BUILD_DIR_DEBUG}
	@cd ${BUILD_DIR_DEBUG} && cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug ../core && make

test: build
	@cd ./$(BUILD_DIR) && ./pptree-test

test-debug: build-debug
	@cd ./$(BUILD_DIR_DEBUG) && ./pptree-test

# Targets for the Dev tools

TOOLS_DIR = .tools

clean-tools:
	@rm -rf ${TOOLS_DIR}

install-tools:
	@mkdir -p ${TOOLS_DIR}
	@cd ${TOOLS_DIR} && cmake ../tools && make

format:
	@cd ${TOOLS_DIR} && make format

format-dry:
	@cd ${TOOLS_DIR} && make format-dry

analyze:
	@cd ${TOOLS_DIR} && make analyze

# Targets for the R package

R_PACKAGE_DIR = bindings/R/PPTree
R_PACKAGE_VERSION = 0.0.0
R_PACKAGE_TARBALL = PPTree_${R_PACKAGE_VERSION}.tar.gz
R_CRAN_MIRROR = https://cran.rstudio.com/

r-install-deps:
	@Rscript -e "install.packages('Rcpp', repos='${R_CRAN_MIRROR}')"
	@Rscript -e "install.packages('RcppEigen', repos='${R_CRAN_MIRROR}')"
	@Rscript -e "install.packages('devtools', repos='${R_CRAN_MIRROR}')"

r-clean:
	@rm -rf \
		${R_PACKAGE_DIR}/src/*.o \
		${R_PACKAGE_DIR}/src/*.so \
		${R_PACKAGE_DIR}/src/*.rds \
		${R_PACKAGE_DIR}/src/*.dll \
		${R_PACKAGE_DIR}/src/core \
		${R_PACKAGE_DIR}/src/.build \
		${R_PACKAGE_DIR}/inst/lib \
		${R_PACKAGE_DIR}/inst/include/nlohmann \
		PPTree_${R_PACKAGE_VERSION}.tar.gzm \
		PPTree.Rcheck

r-document:
	@mkdir -p ${R_PACKAGE_DIR}/src/core && cp -r core/* ${R_PACKAGE_DIR}/src/core
	@cp -r ${NLHOMANN_JSON_HEADERS_PATH}/* ${R_PACKAGE_DIR}/inst/include
	@Rscript -e "devtools::document('${R_PACKAGE_DIR}')"
	@make r-clean

r-build: build r-clean
	@mkdir -p ${R_PACKAGE_DIR}/src/core && cp -r core/* ${R_PACKAGE_DIR}/src/core
	@cp -r ${NLHOMANN_JSON_HEADERS_PATH}/* ${R_PACKAGE_DIR}/inst/include
	@Rscript -e "Rcpp::compileAttributes('${R_PACKAGE_DIR}')"
	@R CMD build ${R_PACKAGE_DIR}
	@make r-clean

r-check: r-build
	@R CMD check ${R_PACKAGE_TARBALL}

r-check-cran: r-build
	@R CMD check ${R_PACKAGE_TARBALL} --as-cran

r-install: r-build
	@R CMD INSTALL ${R_PACKAGE_TARBALL}

r-untar:
	@tar -xvf ${R_PACKAGE_TARBALL}

