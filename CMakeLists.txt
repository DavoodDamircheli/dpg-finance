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
# Hypre — typically installed alongside MFEM inside the container
# ---------------------------------------------------------------------------
find_library(HYPRE_LIB
    NAMES HYPRE
    HINTS /opt/hypre/lib ${MFEM_DIR}/lib /usr/local/lib
    REQUIRED
)
find_path(HYPRE_INCLUDE_DIR
    NAMES HYPRE.h
    HINTS /opt/hypre/include ${MFEM_DIR}/include /usr/local/include
)
message(STATUS "Hypre library : ${HYPRE_LIB}")

# ---------------------------------------------------------------------------
# METIS
# ---------------------------------------------------------------------------
find_library(METIS_LIB
    NAMES metis
    HINTS /usr/local/lib /usr/lib
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
    ${MPI_CXX_INCLUDE_DIRS}
)
target_link_libraries(dpg_common INTERFACE
    ${MFEM_LIB}
    ${HYPRE_LIB}
    ${METIS_LIB}
    MPI::MPI_CXX
)
target_compile_options(dpg_common INTERFACE -O2 -Wall -Wextra)

# ---------------------------------------------------------------------------
# Helper: register a main executable from src/main_*.cpp
# ---------------------------------------------------------------------------
function(add_dpg_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE dpg_common)
endfunction()

# ---------------------------------------------------------------------------
# Executables (one per formulation / version)
# ---------------------------------------------------------------------------
add_dpg_executable(main_european_1d_primal
    src/main_european_1d_primal.cpp)

add_dpg_executable(main_european_1d_ultraweak_mpi
    src/main_european_1d_ultraweak_mpi.cpp)

add_dpg_executable(main_american_1d
    src/main_american_1d.cpp)

add_dpg_executable(main_european_2d_basket_mpi
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

add_dpg_test(test_convection_sign    tests/test_convection_sign.cpp)
add_dpg_test(test_payoff             tests/test_payoff.cpp)
add_dpg_test(test_exact_solution     tests/test_exact_solution.cpp)
add_dpg_test(test_primal_convergence tests/test_primal_convergence.cpp)
add_dpg_test(test_ultraweak_convergence tests/test_ultraweak_convergence.cpp)
add_dpg_test(test_active_set         tests/test_active_set.cpp)
add_dpg_test(test_basket_payoff      tests/test_basket_payoff.cpp)
