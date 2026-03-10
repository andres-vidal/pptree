MAKEFLAGS += --no-print-directory

BUILD_DIR = .build
BUILD_DIR_DEBUG = .debug

NLHOMANN_JSON_HEADERS_PATH = ${BUILD_DIR}/_deps/json-src/include
PCG_HEADERS_PATH = ${BUILD_DIR}/_deps/pcg-src/include

# In CI, disable -march=native to avoid mismatches between cached deps
# and the current runner's CPU instruction set.
CMAKE_EXTRA ?=
ifdef CI
CMAKE_EXTRA += -DPPTREE_NATIVE_ARCH=OFF
endif

clean:
	@rm -rf ${BUILD_DIR} ${BUILD_DIR_DEBUG}

fetch-deps:
	@mkdir -p ${BUILD_DIR}
	@cd ${BUILD_DIR} && cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA} ../core

build: fetch-deps
	@cd ${BUILD_DIR} && make

build-debug:
	@mkdir -p ${BUILD_DIR_DEBUG}
	@cd ${BUILD_DIR_DEBUG} && cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug ${CMAKE_EXTRA} ../core && make

test: build
	@cd ./$(BUILD_DIR) && ./pptree-test

test-debug: build-debug
	@cd ./$(BUILD_DIR_DEBUG) && ./pptree-test

golden-regen: build
	@cd ./$(BUILD_DIR) && ./pptree-golden-gen

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
	@Rscript -e "install.packages('Rcpp', repos='${R_CRAN_MIRROR}', dependencies=TRUE)"
	@Rscript -e "install.packages('RcppEigen', repos='${R_CRAN_MIRROR}', dependencies=TRUE)"
	@Rscript -e "install.packages('devtools', repos='${R_CRAN_MIRROR}', dependencies=TRUE)"
	@Rscript -e "install.packages('jsonlite', repos='${R_CRAN_MIRROR}', dependencies=TRUE)"
	@Rscript -e "install.packages('parsnip', repos='${R_CRAN_MIRROR}', dependencies=TRUE)"
	@Rscript -e "install.packages('pkgdown', repos='${R_CRAN_MIRROR}', dependencies=TRUE)"

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
		${R_PACKAGE_DIR}/inst/include/pcg_* \
		${R_PACKAGE_DIR}/inst/golden \
		PPTree_${R_PACKAGE_VERSION}.tar.gzm \
		PPTree.Rcheck

r-prepare: fetch-deps r-clean
	@mkdir -p ${R_PACKAGE_DIR}/src/core && cp -r core/* ${R_PACKAGE_DIR}/src/core
	@cp -r ${NLHOMANN_JSON_HEADERS_PATH}/* ${R_PACKAGE_DIR}/inst/include
	@cp -r ${PCG_HEADERS_PATH}/* ${R_PACKAGE_DIR}/inst/include
	@cp -r golden ${R_PACKAGE_DIR}/inst/golden

r-document:
	@make r-prepare
	@Rscript -e "devtools::document('${R_PACKAGE_DIR}')"
	@make r-clean

r-build: r-clean
	@make r-prepare
	@rm ${R_PACKAGE_DIR}/src/.core
	@Rscript -e "Rcpp::compileAttributes('${R_PACKAGE_DIR}')"
	@R CMD build ${R_PACKAGE_DIR}
	@touch ${R_PACKAGE_DIR}/src/.core
	@make r-clean

r-check: r-build
	@R CMD check ${R_PACKAGE_TARBALL}

r-check-cran: r-build
	@R CMD check ${R_PACKAGE_TARBALL} --as-cran

r-install: r-build
	@R CMD INSTALL ${R_PACKAGE_TARBALL}

r-untar:
	@tar -xvf ${R_PACKAGE_TARBALL}

# Documentation

DOCS_DIR = docs
DOCS_BUILD_DIR = ${DOCS_DIR}/.build
DOXYGEN = ${TOOLS_DIR}/doxygen/bin/doxygen

docs-site:
	@mkdir -p ${DOCS_BUILD_DIR}
	@cp ${DOCS_DIR}/index.html ${DOCS_DIR}/style.css ${DOCS_BUILD_DIR}/

docs-cpp:
	@mkdir -p ${DOCS_BUILD_DIR}/cpp
	@${DOXYGEN} ${DOCS_DIR}/Doxyfile

docs-r:
	@make r-prepare
	@cd ${BUILD_DIR} && make pptree-core
	@mkdir -p ${R_PACKAGE_DIR}/inst/lib
	@cp ${BUILD_DIR}/libpptree-core.a ${R_PACKAGE_DIR}/inst/lib/
	@cp ${DOCS_DIR}/_pkgdown.yml ${R_PACKAGE_DIR}/_pkgdown.yml
	@Rscript -e "pkgdown::build_site('${R_PACKAGE_DIR}', override=list(destination='../../../${DOCS_BUILD_DIR}/r'), preview=FALSE)"
	@rm -f ${R_PACKAGE_DIR}/_pkgdown.yml
	@make r-clean

docs: docs-site docs-cpp docs-r

# Benchmarking

BENCH_SCENARIOS = bench/default-scenarios.json
BENCH_REF ?= main

benchmark: build
	@${BUILD_DIR}/pptree benchmark -s ${BENCH_SCENARIOS}

benchmark-save: build
	@${BUILD_DIR}/pptree benchmark -s ${BENCH_SCENARIOS} -o bench/results.json --csv bench/results.csv

benchmark-compare: build
	@${BUILD_DIR}/pptree benchmark -s ${BENCH_SCENARIOS} -b bench/results.json

benchmark-vs: build
	@echo "Building and benchmarking current branch..."
	@${BUILD_DIR}/pptree benchmark -s ${BENCH_SCENARIOS} -o bench/.current-results.json -q
	@echo "Setting up baseline (${BENCH_REF})..."
	@git worktree add -f .bench-worktree ${BENCH_REF} 2>/dev/null || { echo "Error: Could not create worktree for ref '${BENCH_REF}'"; exit 1; }
	@mkdir -p .bench-worktree/bench
	@cp ${BENCH_SCENARIOS} .bench-worktree/bench/default-scenarios.json 2>/dev/null || true
	@echo "Building baseline..."
	@cd .bench-worktree && mkdir -p .build && cd .build && cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ../core > /dev/null 2>&1 && make -j > /dev/null 2>&1
	@echo "Running baseline benchmarks..."
	@cd .bench-worktree && .build/pptree benchmark -s bench/default-scenarios.json -o bench/.baseline-results.json -q
	@echo ""
	@${BUILD_DIR}/pptree benchmark -s ${BENCH_SCENARIOS} -b .bench-worktree/bench/.baseline-results.json
	@rm -f bench/.current-results.json
	@git worktree remove -f .bench-worktree 2>/dev/null || true

# Profiling

PROFILE_OUTPUT = pptree-profile.trace
PROFILE_OUTPUT_DEBUG = pptree-profile-debug.trace


profile: build
	@rm -rf ${PROFILE_OUTPUT}
	@xcrun xctrace record --template 'Time Profiler' --output ${PROFILE_OUTPUT} --launch ${BUILD_DIR}/pptree-profile 12 2400 2 10 0.8 10

profile-debug: build-debug
	@rm -rf ${PROFILE_OUTPUT_DEBUG}
	@xcrun xctrace record --template 'Time Profiler' --output ${PROFILE_OUTPUT_DEBUG} --launch ${BUILD_DIR_DEBUG}/pptree-profile 100 100 2 1 1
