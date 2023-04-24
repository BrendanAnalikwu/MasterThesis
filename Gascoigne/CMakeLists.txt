cmake_minimum_required(VERSION 3.22)
project(HeatProblem CXX C)

set(CMAKE_CXX_STANDARD 17)
SET(GASCOIGNE_DIR   "../../../Thesis/GAS/gascoignelib" CACHE PATH "Path to Gascoigne Library")
SET(GASCOIGNE_LIB_DIR "${CMAKE_BINARY_DIR}/../../../../Thesis/GAS/gascoignelib/lib"       CACHE PATH "Path to Gascoigne Library")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "."  CACHE PATH "Path for the executables")
OPTION(MATRIX_DOUBLE_PRECISION "Matrix in double precision" ON)
LINK_DIRECTORIES(${GASCOIGNE_LIB_DIR})
SET(GASCOIGNE_LIBRARY GascoigneStd)

LIST(APPEND GASCOIGNE_LIBRARIES ${GASCOIGNE_LIBRARY})
LIST(APPEND GASCOIGNE_LIBRARIES ${METIS_LIBRARIES})

SET(GASCOIGNE_INCLUDE_DIRS
        ${GASCOIGNE_DIR}/src/Common
        ${GASCOIGNE_DIR}/src/DofHandler
        ${GASCOIGNE_DIR}/src/Discretization/Q1
        ${GASCOIGNE_DIR}/src/Discretization/Q2
        ${GASCOIGNE_DIR}/src/Discretization/DG
        ${GASCOIGNE_DIR}/src/Interface
        ${GASCOIGNE_DIR}/src/LinAlg
        ${GASCOIGNE_DIR}/src/Mesh
        ${GASCOIGNE_DIR}/src/Problems
        ${GASCOIGNE_DIR}/src/Solver
        )
INCLUDE_DIRECTORIES(${GASCOIGNE_INCLUDE_DIRS})

SET(GASCOIGNESTEPBYSTEP_CXX_FLAGS
        "-std=c++11 -pedantic -Wall -Wno-sign-compare -Wno-old-style-cast" CACHE STRING "Configuration flags"
        )
SEPARATE_ARGUMENTS(GASCOIGNESTEPBYSTEP_CXX_FLAGS)
ADD_COMPILE_OPTIONS(${GASCOIGNESTEPBYSTEP_CXX_FLAGS})

LINK_LIBRARIES(${GASCOIGNE_LIBRARIES})
add_executable("HeatProblem" main_heat.cpp heatproblem.h)
add_executable("SeaIceProblem" main_seaice.cpp seaiceproblem.h)
