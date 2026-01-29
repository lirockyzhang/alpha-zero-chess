"""Cython MCTS implementation.

This module provides a Cython-optimized MCTS implementation that is
~5-10x faster than the pure Python version.

Build with:
    cd alpha-zero-chess
    python -m pip install cython
    python alphazero/mcts/cython/setup.py build_ext --inplace

Or using the project's build system:
    uv pip install cython
    uv run python alphazero/mcts/cython/setup.py build_ext --inplace
"""

try:
    from .node import CythonMCTSNode
    from .search import CythonMCTS

    __all__ = ["CythonMCTSNode", "CythonMCTS"]

except ImportError as e:
    import sys
    _import_error = e

    # Provide helpful error message
    def _raise_import_error(*args, **kwargs):
        raise ImportError(
            f"Cython MCTS backend not built. Original error: {_import_error}\n"
            "To build, run:\n"
            "  cd alpha-zero-chess\n"
            "  uv pip install cython\n"
            "  uv run python alphazero/mcts/cython/setup.py build_ext --inplace"
        )

    # Create placeholder classes that raise helpful errors
    class CythonMCTSNode:
        def __init__(self, *args, **kwargs):
            _raise_import_error()

    class CythonMCTS:
        def __init__(self, *args, **kwargs):
            _raise_import_error()

    __all__ = ["CythonMCTSNode", "CythonMCTS"]
