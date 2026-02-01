cmake_minimum_required(VERSION 3.15)
project(alphazero_cpp VERSION 1.0.0 LANGUAGES CXX)

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

# Perft test executable (old implementation - has bugs)
add_executable(perft_test
    tests/perft_test.cpp
)

target_link_libraries(perft_test PRIVATE
    chess_engine
)

# Perft test with chess-library (battle-tested, 269-449M nps)
add_executable(perft_test_library
    tests/perft_test_library.cpp
)

target_include_directories(perft_test_library PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/chess-library/include
)

target_link_libraries(perft_test_library PRIVATE
    Threads::Threads
)

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
    src/mcts/batch_search.cpp
)

target_include_directories(mcts_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/chess-library/include
)

target_link_libraries(mcts_core PUBLIC
    Threads::Threads
    encoding
)

# MCTS test executable
add_executable(mcts_test
    tests/mcts_test.cpp
)

target_link_libraries(mcts_test PRIVATE
    mcts_core
)

# MCTS correctness test executable
add_executable(mcts_correctness_test
    tests/mcts_correctness_test.cpp
)

target_link_libraries(mcts_correctness_test PRIVATE
    mcts_core
)

# MCTS benchmark executable
add_executable(mcts_benchmark
    tests/mcts_benchmark.cpp
)

target_link_libraries(mcts_benchmark PRIVATE
    mcts_core
)

# Enable testing
enable_testing()
add_test(NAME perft_test COMMAND perft_test)
add_test(NAME perft_test_library COMMAND perft_test_library)
add_test(NAME mcts_test COMMAND mcts_test)
add_test(NAME mcts_correctness_test COMMAND mcts_correctness_test)
add_test(NAME mcts_benchmark COMMAND mcts_benchmark)

# Python bindings with pybind11
find_package(Python COMPONENTS Interpreter Development)
if(Python_FOUND)
    message(STATUS "Found Python: ${Python_EXECUTABLE}")

    # Set pybind11 CMake directory
    set(pybind11_DIR "C:/veighna_studio/Lib/site-packages/pybind11/share/cmake/pybind11")

    find_package(pybind11 CONFIG)

    if(pybind11_FOUND)
        message(STATUS "Found pybind11: ${pybind11_DIR}")

        pybind11_add_module(alphazero_cpp
            src/bindings/python_bindings.cpp
        )

        target_link_libraries(alphazero_cpp PRIVATE
            mcts_core
            encoding
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
