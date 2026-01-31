"""MCTS module with multiple backend support.

Available backends:
- python: Pure Python implementation (educational, readable)
- cython: Cython-optimized implementation (5-10x faster)
- cpp: C++ implementation with pybind11 (20-50x faster)
"""

from typing import Optional

from .base import MCTSBase, MCTSNodeBase, MCTSStats
from .evaluator import NetworkEvaluator, RandomEvaluator, CachedEvaluator
from ..config import MCTSConfig, MCTSBackend


def create_mcts(
    backend: MCTSBackend = MCTSBackend.PYTHON,
    config: Optional[MCTSConfig] = None
) -> MCTSBase:
    """Factory function to create MCTS with specified backend.

    Args:
        backend: Which MCTS implementation to use
        config: MCTS configuration

    Returns:
        MCTS instance
    """
    if config is None:
        config = MCTSConfig()

    if backend == MCTSBackend.PYTHON:
        from .python.search import PythonMCTS
        return PythonMCTS(config)

    elif backend == MCTSBackend.CYTHON:
        try:
            from .cython.search import CythonMCTS
            return CythonMCTS(config)
        except ImportError as e:
            raise ImportError(
                "Cython MCTS backend not available. "
                "Build it with: python setup.py build_ext --inplace"
            ) from e

    elif backend == MCTSBackend.CPP:
        try:
            from .cpp import CppMCTS
            return CppMCTS(config)
        except ImportError as e:
            raise ImportError(
                "C++ MCTS backend not available. "
                "Build it with: cmake --build build"
            ) from e

    else:
        raise ValueError(f"Unknown MCTS backend: {backend}")


def get_available_backends():
    """Get list of available MCTS backends."""
    available = [MCTSBackend.PYTHON]

    try:
        from .cython.search import CythonMCTS
        available.append(MCTSBackend.CYTHON)
    except ImportError:
        pass

    try:
        from .cpp import CppMCTS
        available.append(MCTSBackend.CPP)
    except ImportError:
        pass

    return available


def get_best_backend() -> MCTSBackend:
    """Auto-detect the best available MCTS backend.

    Priority order (fastest to slowest):
    1. C++ (pybind11) - 20-50x faster than Python
    2. Cython - 5-10x faster than Python
    3. Python - baseline (always available)

    Returns:
        MCTSBackend enum value for the fastest available backend
    """
    # Try C++ first (fastest)
    try:
        from .cpp import CppMCTS
        return MCTSBackend.CPP
    except ImportError:
        pass

    # Try Cython second
    try:
        from .cython.search import CythonMCTS
        return MCTSBackend.CYTHON
    except ImportError:
        pass

    # Fall back to Python (always available)
    return MCTSBackend.PYTHON


__all__ = [
    "MCTSBase",
    "MCTSNodeBase",
    "MCTSStats",
    "MCTSBackend",
    "MCTSConfig",
    "NetworkEvaluator",
    "RandomEvaluator",
    "CachedEvaluator",
    "create_mcts",
    "get_available_backends",
    "get_best_backend",
]
