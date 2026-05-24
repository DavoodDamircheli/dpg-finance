cmake_minimum_required(VERSION 3.16)
project(dpg-finance LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------------------
# MPI (MPICH inside the container)
# ---------------------------------------------------------------------------
find_package(MPI REQUIRED COMPONENTS CXX)

# ---------------------------------------------------------------------------
# MFEM — installed at /usr/local inside the container
# ---------------------------------------------------------------------------
set(MFEM_DIR /usr/local CACHE PATH "Root of the MFEM installation")

find_library(MFEM_LIB
    NAMES mfem
    HINTS ${MFEM_DIR}/lib
    REQUIRED
)
find_path(MFEM_INCLUDE_DIR
    NAMES mfem.hpp
    HINTS ${MFEM_DIR}/include
    REQUIRED
)
message(STATUS "MFEM library : ${MFEM_LIB}")
message(STATUS "MFEM includes: ${MFEM_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# Hypre — at /opt/hypre inside the container
# MFEM's headers #include <HYPRE_utilities.h> (angle brackets), so
# /opt/hypre/include MUST be on the include path.
# ---------------------------------------------------------------------------
set(HYPRE_ROOT /opt/hypre CACHE PATH "Hypre installation root")
find_library(HYPRE_LIB
    NAMES HYPRE
    HINTS ${HYPRE_ROOT}/lib /usr/local/lib
    REQUIRED
)
find_path(HYPRE_INCLUDE_DIR
    NAMES HYPRE_utilities.h
    HINTS ${HYPRE_ROOT}/include
    REQUIRED
)
message(STATUS "Hypre library : ${HYPRE_LIB}")
message(STATUS "Hypre includes: ${HYPRE_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# SuiteSparse (UMFPack, KLU, AMD …) — MFEM links against these
# ---------------------------------------------------------------------------
find_path(SUITESPARSE_INCLUDE_DIR
    NAMES umfpack.h
    HINTS /usr/include/suitesparse /usr/include
)
find_library(UMFPACK_LIB  NAMES umfpack  HINTS /usr/lib/x86_64-linux-gnu)
find_library(KLU_LIB      NAMES klu      HINTS /usr/lib/x86_64-linux-gnu)
find_library(AMD_LIB      NAMES amd      HINTS /usr/lib/x86_64-linux-gnu)
find_library(CHOLMOD_LIB  NAMES cholmod  HINTS /usr/lib/x86_64-linux-gnu)
find_library(COLAMD_LIB   NAMES colamd  HINTS /usr/lib/x86_64-linux-gnu)
find_library(CAMD_LIB     NAMES camd    HINTS /usr/lib/x86_64-linux-gnu)
find_library(CCOLAMD_LIB  NAMES ccolamd HINTS /usr/lib/x86_64-linux-gnu)
find_library(SS_CONFIG_LIB NAMES suitesparseconfig HINTS /usr/lib/x86_64-linux-gnu)
find_library(OPENBLAS_LIB  NAMES openblas  HINTS /usr/lib/x86_64-linux-gnu)

set(SUITESPARSE_LIBS
    ${UMFPACK_LIB} ${KLU_LIB} ${AMD_LIB} ${CHOLMOD_LIB}
    ${COLAMD_LIB} ${CAMD_LIB} ${CCOLAMD_LIB}
    ${SS_CONFIG_LIB} ${OPENBLAS_LIB})

# ---------------------------------------------------------------------------
# METIS
# ---------------------------------------------------------------------------
find_library(METIS_LIB
    NAMES metis
    HINTS /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib
    REQUIRED
)
message(STATUS "METIS library : ${METIS_LIB}")

# ---------------------------------------------------------------------------
# Common interface target shared by every executable and test
# ---------------------------------------------------------------------------
add_library(dpg_common INTERFACE)
target_include_directories(dpg_common INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${MFEM_INCLUDE_DIR}
    ${HYPRE_INCLUDE_DIR}
    /usr/include
    /usr/include/suitesparse
    ${MPI_CXX_INCLUDE_DIRS}
)
target_link_libraries(dpg_common INTERFACE
    ${MFEM_LIB}
    ${HYPRE_LIB}
    ${SUITESPARSE_LIBS}
    ${METIS_LIB}
    MPI::MPI_CXX
)
target_compile_options(dpg_common INTERFACE -O2 -Wall -Wextra)

# ---------------------------------------------------------------------------
# DPG utility library — pre-built .o files from the MFEM miniapp build
# (needed by the ultraweak DPG formulation: ParDPGWeakForm, etc.)
# ---------------------------------------------------------------------------
set(DPG_UTIL_DIR
    /opt/mfem/artifacts/miniapps/dpg/CMakeFiles/pconvection-diffusion-dpg.dir/util)

add_library(dpg_util INTERFACE)
target_sources(dpg_util INTERFACE
    ${DPG_UTIL_DIR}/pweakform.cpp.o
    ${DPG_UTIL_DIR}/weakform.cpp.o
    ${DPG_UTIL_DIR}/blockstaticcond.cpp.o
    ${DPG_UTIL_DIR}/complexstaticcond.cpp.o
    ${DPG_UTIL_DIR}/complexweakform.cpp.o
    ${DPG_UTIL_DIR}/pcomplexweakform.cpp.o
    ${DPG_UTIL_DIR}/pml.cpp.o
)
target_include_directories(dpg_util INTERFACE /opt/mfem/miniapps/dpg)

# ---------------------------------------------------------------------------
# Helper: register a main executable from src/main_*.cpp
# ---------------------------------------------------------------------------
function(add_dpg_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE dpg_common)
endfunction()

# Helper for executables that need the DPG ultraweak (ParDPGWeakForm) utilities
function(add_dpg_uw_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE dpg_common dpg_util)
endfunction()

# ---------------------------------------------------------------------------
# Executables (one per formulation / version)
# ---------------------------------------------------------------------------
add_dpg_executable(main_european_1d_primal
    src/main_european_1d_primal.cpp)

add_dpg_uw_executable(main_european_1d_ultraweak_mpi
    src/main_european_1d_ultraweak_mpi.cpp)

add_dpg_executable(main_american_1d
    src/main_american_1d.cpp)

add_dpg_uw_executable(main_european_2d_basket_mpi
    src/main_european_2d_basket_mpi.cpp)

add_dpg_executable(main_barrier_1d
    src/main_barrier_1d.cpp)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_TESTS ${CMAKE_BINARY_DIR}/tests)

function(add_dpg_test name source)
    add_executable(${name} ${source})
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    target_link_libraries(${name} PRIVATE dpg_common)
    add_test(NAME ${name} COMMAND ${CMAKE_BINARY_DIR}/tests/${name})
endfunction()

# Helper for tests that need ParDPGWeakForm utilities
function(add_dpg_uw_test name source)
    add_executable(${name} ${source})
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    target_link_libraries(${name} PRIVATE dpg_common dpg_util)
    add_test(NAME ${name} COMMAND ${CMAKE_BINARY_DIR}/tests/${name})
endfunction()

add_dpg_test(test_convection_sign    tests/test_convection_sign.cpp)
add_dpg_test(test_payoff             tests/test_payoff.cpp)
add_dpg_test(test_exact_solution     tests/test_exact_solution.cpp)
add_dpg_test(test_primal_convergence tests/test_primal_convergence.cpp)
add_dpg_uw_test(test_ultraweak_convergence tests/test_ultraweak_convergence.cpp)
add_dpg_test(test_active_set                    tests/test_active_set.cpp)
add_dpg_test(test_american_dominates_european   tests/test_american_dominates_european.cpp)
add_dpg_test(test_basket_payoff      tests/test_basket_payoff.cpp)
add_dpg_uw_test(test_delta_extraction    tests/test_delta_extraction.cpp)
add_dpg_uw_test(test_mpi_consistency    tests/test_mpi_consistency.cpp)
add_dpg_test(test_diffusion_tensor      tests/test_diffusion_tensor.cpp)

# Timeouts (seconds) for tests that run a DPG solve
set_tests_properties(test_ultraweak_convergence PROPERTIES TIMEOUT 300)
set_tests_properties(test_delta_extraction      PROPERTIES TIMEOUT 120)
set_tests_properties(test_mpi_consistency       PROPERTIES TIMEOUT  60)
set_tests_properties(test_diffusion_tensor              PROPERTIES TIMEOUT  30)
set_tests_properties(test_active_set                    PROPERTIES TIMEOUT  30)
set_tests_properties(test_american_dominates_european   PROPERTIES TIMEOUT  60)
