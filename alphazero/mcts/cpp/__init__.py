"""C++ MCTS implementation with pybind11 bindings.

This module provides a C++ MCTS implementation that is ~20-50x faster
than the pure Python version.

Build with:
    cd alpha-zero-chess/alphazero/mcts/cpp
    mkdir build && cd build
    cmake ..
    cmake --build . --config Release
    cmake --install .

Or on Windows with Visual Studio:
    cd alpha-zero-chess/alphazero/mcts/cpp
    mkdir build && cd build
    cmake .. -G "Visual Studio 17 2022"
    cmake --build . --config Release
    cmake --install .

Requirements:
    - CMake 3.15+
    - C++17 compiler (GCC 7+, Clang 5+, MSVC 2019+)
    - pybind11 (pip install pybind11)
"""

try:
    from .mcts_cpp import CppMCTSNode, CppMCTS

    __all__ = ["CppMCTSNode", "CppMCTS"]

except ImportError as e:
    import sys
    _import_error = e

    def _raise_import_error(*args, **kwargs):
        raise ImportError(
            f"C++ MCTS backend not built. Original error: {_import_error}\n"
            "To build, run:\n"
            "  cd alpha-zero-chess/alphazero/mcts/cpp\n"
            "  mkdir build && cd build\n"
            "  cmake ..\n"
            "  cmake --build . --config Release\n"
            "  cmake --install .\n"
            "\n"
            "Requirements:\n"
            "  - CMake 3.15+\n"
            "  - C++17 compiler\n"
            "  - pybind11 (pip install pybind11)"
        )

    class CppMCTSNode:
        def __init__(self, *args, **kwargs):
            _raise_import_error()

    class CppMCTS:
        def __init__(self, *args, **kwargs):
            _raise_import_error()

    __all__ = ["CppMCTSNode", "CppMCTS"]
