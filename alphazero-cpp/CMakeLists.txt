cmake_minimum_required(VERSION 3.15)
project(alphazero_cpp VERSION 1.0.0 LANGUAGES CXX)

# Required for building shared libraries (.so) on Linux
# Static libs linked into pybind11 module need position-independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Try C++20, fallback to C++17
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Check if C++20 is available, fallback to C++17
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
        set(CMAKE_CXX_STANDARD 17)
        message(STATUS "C++20 not available, using C++17")
    else()
        message(STATUS "Using C++20")
    endif()
endif()

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        add_compile_options(/O2 /arch:AVX2)
    else()
        add_compile_options(-O3 -march=native -mavx2 -mpopcnt -mbmi2)
    endif()
endif()

# Find required packages
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# Use ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()

# Chess engine library
add_library(chess_engine STATIC
    src/chess/bitboard.cpp
    src/chess/move.cpp
    src/chess/zobrist.cpp
    src/chess/position.cpp
    src/chess/movegen.cpp
)

target_include_directories(chess_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(chess_engine PUBLIC
    Threads::Threads
)

# NOTE: Old test executables disabled - they use deprecated API
# Keeping files in tests/ for reference but not building them

# Encoding library
add_library(encoding STATIC
    src/encoding/position_encoder.cpp
    src/encoding/move_encoder.cpp
)

target_include_directories(encoding PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/chess-library/include
)

target_link_libraries(encoding PUBLIC
    OpenMP::OpenMP_CXX
)

# MCTS library
add_library(mcts_core STATIC
    src/mcts/search.cpp
    src/mcts/batch_coordinator.cpp
)

target_include_directories(mcts_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/chess-library/include
)

target_link_libraries(mcts_core PUBLIC
    Threads::Threads
    encoding
)

# Self-play library
add_library(selfplay STATIC
    src/selfplay/game.cpp
    src/selfplay/coordinator.cpp
    src/selfplay/evaluation_queue.cpp
    src/selfplay/parallel_coordinator.cpp
)

target_include_directories(selfplay PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/chess-library/include
)

target_link_libraries(selfplay PUBLIC
    mcts_core
    encoding
    training
)

# Training library
add_library(training STATIC
    src/training/replay_buffer.cpp
    src/training/trainer.cpp
)

target_include_directories(training PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(training PUBLIC
    Threads::Threads
)

# NOTE: MCTS test executables disabled - they use deprecated API
# Use alphazero-cpp/scripts/test_performance.py instead for testing

# Python bindings with pybind11
find_package(Python COMPONENTS Interpreter Development)
if(Python_FOUND)
    message(STATUS "Found Python: ${Python_EXECUTABLE}")

    # Set pybind11 CMake directory (use -Dpybind11_DIR=... to override)
    if(NOT DEFINED pybind11_DIR)
        if(WIN32)
            set(pybind11_DIR "C:/Users/liroc/OneDrive - Duke University/2025-2026/projects/alpha-zero-chess/.venv/Lib/site-packages/pybind11/share/cmake/pybind11")
        endif()
    endif()

    find_package(pybind11 CONFIG)

    if(pybind11_FOUND)
        message(STATUS "Found pybind11: ${pybind11_DIR}")

        pybind11_add_module(alphazero_cpp
            src/bindings/python_bindings.cpp
        )

        target_link_libraries(alphazero_cpp PRIVATE
            mcts_core
            encoding
            selfplay
            training
        )

        target_include_directories(alphazero_cpp PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/chess-library/include
        )

        message(STATUS "Python bindings will be built")
    else()
        message(STATUS "pybind11 not found, skipping Python bindings")
    endif()
else()
    message(STATUS "Python not found, skipping Python bindings")
endif()

# Installation
install(TARGETS chess_engine
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)
