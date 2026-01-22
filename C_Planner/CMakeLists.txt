cmake_minimum_required(VERSION 3.10)
project(FrenetOptimalPlanner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find Python first
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find required packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Set pybind11 cmake directory directly
set(pybind11_DIR "/home/wenguang/.cache/pypoetry/virtualenvs/fiss-plus-planner-IsomLo2q-py3.10/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Python3_INCLUDE_DIRS})

option(USE_RECORDER "Use recorder" OFF)
set(recorder_file ${recorder_file})
if (USE_RECORDER)
  MESSAGE(STATUS "USE_RECORDER=ON")
  add_definitions(-DUSE_RECORDER)
  set(recorder_file ${recorder_file} Recorder4Cpp/recorder.h
  	Recorder4Cpp/recorder.cpp)
endif()

# Source files for the common geometry module
set(GEOMETRY_SOURCES
    common/geometry/polynomial.cpp
    common/geometry/cubic_spline.cpp
)

# Source files for the common cost module
set(COST_SOURCES
    common/cost/cost_function.cpp
)

# Source files for the common collision module
set(COLLISION_SOURCES
    common/collision/collision_checker.cpp
)

# Source files for the common utils module
set(UTILS_SOURCES
    common/utils/array_utils.cpp
)

# Source files for the main planner
set(PLANNER_SOURCES
    Frenet_Planner.cpp
    ${GEOMETRY_SOURCES}
    ${COST_SOURCES}
    ${COLLISION_SOURCES}
    ${UTILS_SOURCES}
)

# Source files for FISS+ planner (includes base planner sources)
set(FISS_PLUS_PLANNER_SOURCES
    Fiss_Plus_Planner.cpp
    Frenet_Planner.cpp
    ${GEOMETRY_SOURCES}
    ${COST_SOURCES}
    ${COLLISION_SOURCES}
    ${UTILS_SOURCES}
)

# Create Python extension module using PyBind11 for Frenet Planner
pybind11_add_module(frenet_planner_cpp pybind11_bindings.cpp ${PLANNER_SOURCES} ${recorder_file})
target_link_libraries(frenet_planner_cpp PUBLIC Eigen3::Eigen)

# Create Python extension module using PyBind11 for FISS+ Planner
pybind11_add_module(fiss_plus_planner_cpp fiss_plus_pybind11_bindings.cpp ${FISS_PLUS_PLANNER_SOURCES} ${recorder_file})
target_link_libraries(fiss_plus_planner_cpp PUBLIC Eigen3::Eigen)

# Installation
install(TARGETS frenet_planner_cpp fiss_plus_planner_cpp
        LIBRARY DESTINATION lib)

install(FILES Frenet_Planner.h Fiss_Plus_Planner.h DESTINATION include)
install(DIRECTORY common/ DESTINATION include/common)
