cmake_minimum_required(VERSION 3.15)
project(alphazero_mcts_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    # Try to find pybind11 via pip
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(pybind11_DIR)
        find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})
    else()
        message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
    endif()
endif()

# Create the Python module
pybind11_add_module(mcts_cpp
    src/bindings.cpp
)

target_include_directories(mcts_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Python3_NumPy_INCLUDE_DIRS}
)

# Optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mcts_cpp PRIVATE -O3 -march=native -ffast-math)
elseif(MSVC)
    target_compile_options(mcts_cpp PRIVATE /O2 /fp:fast)
endif()

# Install to the package directory
install(TARGETS mcts_cpp
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
)

# Print configuration
message(STATUS "Python: ${Python3_EXECUTABLE}")
message(STATUS "pybind11: ${pybind11_VERSION}")
message(STATUS "NumPy include: ${Python3_NumPy_INCLUDE_DIRS}")
