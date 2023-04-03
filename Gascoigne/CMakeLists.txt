cmake_minimum_required(VERSION 3.24)
project(SEAICE)

set(CMAKE_CXX_STANDARD 14)
SET(GASCOIGNE_DIR   "../../../Thesis/GAS/gascoignelib")
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

add_executable("SeaIceDynamics" main.cpp)
