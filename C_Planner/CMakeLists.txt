cmake_minimum_required(VERSION 3.10)
project(FrenetOptimalPlanner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python first
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find required packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find pybind11 - try multiple methods
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  # Try to find pybind11 in Python site-packages
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  find_package(pybind11 CONFIG REQUIRED)
endif()

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

# Create Python extension module using PyBind11
pybind11_add_module(frenet_planner_cpp pybind11_bindings.cpp ${PLANNER_SOURCES} ${recorder_file})
target_link_libraries(frenet_planner_cpp PUBLIC Eigen3::Eigen)

# Installation
install(TARGETS frenet_planner_cpp
        LIBRARY DESTINATION lib)

install(FILES Frenet_Planner.h DESTINATION include)
install(DIRECTORY common/ DESTINATION include/common)
