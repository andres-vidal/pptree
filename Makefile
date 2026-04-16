MAKEFLAGS += --no-print-directory

BUILD_DIR = .build
BUILD_DIR_DEBUG = .debug
BUILD_DIR_COV = .coverage
R_BUILD_DIR = .r-build

NLHOMANN_JSON_HEADERS_PATH = ${BUILD_DIR}/_deps/json-src/include
PCG_HEADERS_PATH = ${BUILD_DIR}/_deps/pcg-src/include

CMAKE_EXTRA ?=

clean:
	@rm -rf ${BUILD_DIR} ${BUILD_DIR_DEBUG} ${BUILD_DIR_COV} ${R_BUILD_DIR}

fetch-deps:
	@mkdir -p ${BUILD_DIR}
	@cd ${BUILD_DIR} && cmake -G "Unix Makefiles" \
		-DCMAKE_BUILD_TYPE=Release \
		-DPPFOREST2_CORE_ONLY=ON \
		${CMAKE_EXTRA} ../core

build:
	@mkdir -p ${BUILD_DIR}
	@cd ${BUILD_DIR} && cmake -G "Unix Makefiles" \
		-DCMAKE_BUILD_TYPE=Release -DPPFOREST2_CORE_ONLY=OFF \
		${CMAKE_EXTRA} ../core && make

build-debug:
	@mkdir -p ${BUILD_DIR_DEBUG}
	@cd ${BUILD_DIR_DEBUG} && cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug ${CMAKE_EXTRA} ../core && make

test: build
	@cd ./$(BUILD_DIR) && ./ppforest2-test

test-debug: build-debug
	@cd ./$(BUILD_DIR_DEBUG) && ./ppforest2-test

build-coverage:
	@mkdir -p ${BUILD_DIR_COV}
	@cd ${BUILD_DIR_COV} && cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug -DPPFOREST2_COVERAGE=ON ${CMAKE_EXTRA} ../core && make

test-coverage: build-coverage
	@cd ./$(BUILD_DIR_COV) && ./ppforest2-test

LCOV_IGNORE = --ignore-errors mismatch,inconsistent,unsupported,range,format,category,unused

coverage: test-coverage
	@lcov --capture --directory ${BUILD_DIR_COV} --output-file ${BUILD_DIR_COV}/coverage.info --quiet ${LCOV_IGNORE} --rc branch_coverage=0
	@lcov --extract ${BUILD_DIR_COV}/coverage.info '*/core/src/*' -o ${BUILD_DIR_COV}/coverage-filtered.info --quiet ${LCOV_IGNORE}
	@lcov --remove ${BUILD_DIR_COV}/coverage-filtered.info '*/_deps/*' '*.test.*' '*/golden/*' '*test.cpp' -o ${BUILD_DIR_COV}/coverage-filtered.info --quiet ${LCOV_IGNORE}
	@genhtml ${BUILD_DIR_COV}/coverage-filtered.info -o ${BUILD_DIR_COV}/html --quiet ${LCOV_IGNORE}
	@python3 scripts/coverage-report.py ${BUILD_DIR_COV}/html

golden-regen: build
	@cd ./$(BUILD_DIR) && ./ppforest2-golden-gen

# Dev tools (clang-format, clang-tidy, cppcheck via pip; doxygen via cmake)

TOOLS_DIR = .tools

install-tools:
	@pip install -r requirements-dev.txt
	@command -v asdf >/dev/null 2>&1 && asdf reshim python || true
	@mkdir -p ${TOOLS_DIR}
	@cd ${TOOLS_DIR} && cmake ../tools && make

install-doxygen:
	@mkdir -p ${TOOLS_DIR}
	@cd ${TOOLS_DIR} && cmake ../tools && make

clean-tools:
	@rm -rf ${TOOLS_DIR}

FORMAT_SOURCES = $(shell find core/src core/include -name '*.cpp' -o -name '*.hpp' -o -name '*.h') bindings/R/src/main.cpp bindings/R/inst/include/ppforest2.h
TIDY_SOURCES = $(shell find core/src -name '*.cpp' ! -name '*.test.cpp')

format:
	@clang-format -i ${FORMAT_SOURCES}

format-dry:
	@clang-format --dry-run --Werror ${FORMAT_SOURCES}

tidy: build
	@echo ${TIDY_SOURCES} | tr ' ' '\n' | xargs -P $$(nproc 2>/dev/null || sysctl -n hw.ncpu) -n 1 clang-tidy -p ${BUILD_DIR}

analyze:
	@cppcheck --enable=all --check-level=exhaustive --suppress=missingIncludeSystem --suppress=duplInheritedMember --quiet core -Icore/src -Icore/include

CPPCLEAN = $(shell python3 -c "import sysconfig; print(sysconfig.get_path('scripts'))")/cppclean

cppclean: build
	@${CPPCLEAN} \
		--include-path=core/src \
		--include-path=core/include \
		--include-path=${BUILD_DIR}/_deps/eigen-src \
		--include-path=${BUILD_DIR}/_deps/json-src/include \
		--include-path=${BUILD_DIR}/_deps/pcg-src/include \
		--include-path=${BUILD_DIR}/_deps/csv-src/include \
		--include-path=${BUILD_DIR}/_deps/fmt-src/include \
		--include-path=${BUILD_DIR}/_deps/cli11-src/include \
		--include-path=${BUILD_DIR}/_deps/googletest-src/googletest/include \
		core/src 2>&1

# Targets for the R package

R_PACKAGE_DIR = bindings/R
CORE_VERSION := $(shell cat VERSION)
R_PACKAGE_TARBALL = ppforest2_${CORE_VERSION}.tar.gz
R_CRAN_MIRROR = https://cran.rstudio.com/

r-install-deps:
	@Rscript -e "if (!requireNamespace('pak', quietly = TRUE)) install.packages('pak', repos = '${R_CRAN_MIRROR}')"
	@Rscript -e "pak::local_install_deps('${R_PACKAGE_DIR}')"

r-build-core: fetch-deps
	@mkdir -p ${R_BUILD_DIR}/_deps
	@for src_dir in ${BUILD_DIR}/_deps/*-src; do \
		target=${R_BUILD_DIR}/_deps/$$(basename $$src_dir); \
		[ ! -d "$$target" ] && cp -r "$$src_dir" "$$target" || true; \
	done
	@R_CXX_FULL="$$(R CMD config CXX17)"; \
	R_CXX_COMPILER="$$(echo $$R_CXX_FULL | awk '{print $$1}')"; \
	R_CXX_EXTRA="$$(echo $$R_CXX_FULL | awk '{$$1=""; print}' | sed 's/^ *//')"; \
	cd ${R_BUILD_DIR} && cmake -G "Unix Makefiles" \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CXX_COMPILER="$$R_CXX_COMPILER" \
		-DCMAKE_CXX_FLAGS="$$R_CXX_EXTRA $$(R CMD config CXX17STD) $$(R CMD config CXX17FLAGS)" \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DPPFOREST2_CORE_ONLY=ON ${CMAKE_EXTRA} \
		../core
	@cd ${R_BUILD_DIR} && make ppforest2-core
	@strip -S ${R_BUILD_DIR}/libppforest2-core.a

r-clean:
	@rm -rf \
		${R_PACKAGE_DIR}/src/*.o \
		${R_PACKAGE_DIR}/src/*.so \
		${R_PACKAGE_DIR}/src/*.rds \
		${R_PACKAGE_DIR}/src/*.dll \
		${R_PACKAGE_DIR}/src/core \
		${R_PACKAGE_DIR}/src/.build \
		${R_PACKAGE_DIR}/src/VERSION \
		${R_PACKAGE_DIR}/NEWS.md \
		${R_PACKAGE_DIR}/inst/lib \
		${R_PACKAGE_DIR}/inst/include/nlohmann \
		${R_PACKAGE_DIR}/inst/include/pcg_* \
		${R_PACKAGE_DIR}/inst/golden \
		ppforest2_${CORE_VERSION}.tar.gzm \
		ppforest2.Rcheck

r-version:
	@sed -i.bak 's/^Version: .*/Version: ${CORE_VERSION}/' ${R_PACKAGE_DIR}/DESCRIPTION && rm -f ${R_PACKAGE_DIR}/DESCRIPTION.bak
	@sed -i.bak "s/^Date: .*/Date: $$(date +%Y-%m-%d)/" ${R_PACKAGE_DIR}/DESCRIPTION && rm -f ${R_PACKAGE_DIR}/DESCRIPTION.bak
	
	
r-prepare: r-clean r-version fetch-deps
	@mkdir -p ${R_PACKAGE_DIR}/src/core && cp -r core/* ${R_PACKAGE_DIR}/src/core
	@cp VERSION ${R_PACKAGE_DIR}/src/VERSION
	@cp CHANGELOG.md ${R_PACKAGE_DIR}/NEWS.md
	@cp -r ${NLHOMANN_JSON_HEADERS_PATH}/* ${R_PACKAGE_DIR}/inst/include
	@cp -r ${PCG_HEADERS_PATH}/* ${R_PACKAGE_DIR}/inst/include
	@cp -r golden ${R_PACKAGE_DIR}/inst/golden

r-document:
	@make r-prepare
	@Rscript -e "devtools::document('${R_PACKAGE_DIR}')"
	@make r-clean

r-build: r-clean
	@make r-prepare
	@Rscript -e "Rcpp::compileAttributes('${R_PACKAGE_DIR}')"
	@R CMD build ${R_PACKAGE_DIR}
	@make r-clean

r-test:
	@make r-prepare
	@Rscript -e "Rcpp::compileAttributes('${R_PACKAGE_DIR}')"
	@PPFOREST2_FETCH_CACHE="$(CURDIR)/${BUILD_DIR}/_deps" Rscript -e "devtools::load_all('${R_PACKAGE_DIR}'); devtools::test('${R_PACKAGE_DIR}')"
	@make r-clean

r-check: r-build
	@PPFOREST2_FETCH_CACHE="$(CURDIR)/${BUILD_DIR}/_deps" R CMD check ${R_PACKAGE_TARBALL}

r-check-cran: r-build
	@PPFOREST2_FETCH_CACHE="$(CURDIR)/${BUILD_DIR}/_deps" R CMD check ${R_PACKAGE_TARBALL} --as-cran

r-install: r-build
	@PPFOREST2_FETCH_CACHE="$(CURDIR)/${BUILD_DIR}/_deps" R CMD INSTALL ${R_PACKAGE_TARBALL}

# Documentation

DOCS_DIR = docs
DOCS_BUILD_DIR = ${DOCS_DIR}/.build
DOXYGEN = ${TOOLS_DIR}/doxygen/bin/doxygen
DOCS_REF ?= main

docs-site:
	@mkdir -p ${DOCS_BUILD_DIR}
	@sed 's/{{VERSION}}/v${CORE_VERSION}/g' ${DOCS_DIR}/index.html > ${DOCS_BUILD_DIR}/index.html
	@cp ${DOCS_DIR}/style.css ${DOCS_BUILD_DIR}/

docs-cpp:
	@mkdir -p ${DOCS_BUILD_DIR}/cpp
	@( cat ${DOCS_DIR}/Doxyfile ; echo "PROJECT_NUMBER = v${CORE_VERSION}" ) | ${DOXYGEN} -

docs-r:
	@make r-build-core
	@make r-prepare
	@mkdir -p ${R_PACKAGE_DIR}/inst/lib
	@cp ${R_BUILD_DIR}/libppforest2-core.a ${R_PACKAGE_DIR}/inst/lib/
	@cp ${DOCS_DIR}/_pkgdown.yml ${R_PACKAGE_DIR}/_pkgdown.yml
	@sed -i.bak 's|/ppforest2/main/|/ppforest2/${DOCS_REF}/|g' ${R_PACKAGE_DIR}/_pkgdown.yml ${R_PACKAGE_DIR}/README.md && rm -f ${R_PACKAGE_DIR}/_pkgdown.yml.bak ${R_PACKAGE_DIR}/README.md.bak
	@Rscript -e "pkgdown::build_site('${R_PACKAGE_DIR}', override=list(destination='../../${DOCS_BUILD_DIR}/r'), preview=FALSE)"
	@rm -f ${R_PACKAGE_DIR}/_pkgdown.yml
	@make r-clean

docs: docs-site docs-cpp docs-r

# Release management

RELEASE_TAG ?= v${CORE_VERSION}

release:
	@git tag -a ${RELEASE_TAG} -m "Release ${RELEASE_TAG}"
	@git push origin ${RELEASE_TAG}

release-revert:
	@echo "This will delete the local and remote git tag '${RELEASE_TAG}'."
	@echo "Press Enter to continue or Ctrl-C to abort." && read _
	git tag -d ${RELEASE_TAG}
	git push origin :refs/tags/${RELEASE_TAG}
	@echo "Reverted release ${RELEASE_TAG}."
	@echo "Note: if a GitHub Release exists for this tag, delete it manually at"
	@echo "  https://github.com/$$(git remote get-url origin | sed 's|.*github.com[:/]||;s|\.git$$||')/releases"

# Benchmarking

BENCH_SCENARIOS = bench/default-scenarios.json
BENCH_REF ?= main

benchmark: build
	@${BUILD_DIR}/ppforest2 benchmark -s ${BENCH_SCENARIOS}

benchmark-save: build
	@${BUILD_DIR}/ppforest2 benchmark -s ${BENCH_SCENARIOS} -o bench/results.json -o bench/results.csv

benchmark-compare: build
	@${BUILD_DIR}/ppforest2 benchmark -s ${BENCH_SCENARIOS} -b bench/results.json

benchmark-vs: build
	@echo "Building and benchmarking current branch..."
	@${BUILD_DIR}/ppforest2 benchmark -s ${BENCH_SCENARIOS} -o bench/.current-results.json -q
	@echo "Setting up baseline (${BENCH_REF})..."
	@git worktree add -f .bench-worktree ${BENCH_REF} 2>/dev/null || { echo "Error: Could not create worktree for ref '${BENCH_REF}'"; exit 1; }
	@mkdir -p .bench-worktree/bench
	@cp ${BENCH_SCENARIOS} .bench-worktree/bench/default-scenarios.json 2>/dev/null || true
	@echo "Building baseline..."
	@cd .bench-worktree && mkdir -p .build && cd .build && cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ../core > /dev/null 2>&1 && make -j > /dev/null 2>&1
	@echo "Running baseline benchmarks..."
	@cd .bench-worktree && .build/ppforest2 benchmark -s bench/default-scenarios.json -o bench/.baseline-results.json -q
	@echo ""
	@${BUILD_DIR}/ppforest2 benchmark -s ${BENCH_SCENARIOS} -b .bench-worktree/bench/.baseline-results.json
	@rm -f bench/.current-results.json
	@git worktree remove -f .bench-worktree 2>/dev/null || true