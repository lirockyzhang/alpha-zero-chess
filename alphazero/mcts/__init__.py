"""MCTS module with multiple backend support.

Available backends:
- python: Pure Python implementation (educational, readable)
- cython: Cython-optimized implementation (5-10x faster)
- cpp: C++ implementation with pybind11 (20-50x faster)

C++ Backend Options:
- CppBatchedMCTS: Proper AlphaZero with batch leaf evaluation (default, recommended for training)
- CppMCTS: Fast but root-only evaluation (use for inference/play only)
"""

from typing import Optional

from .base import MCTSBase, MCTSNodeBase, MCTSStats
from .evaluator import NetworkEvaluator, RandomEvaluator, CachedEvaluator
from ..config import MCTSConfig, MCTSBackend


def create_mcts(
    backend: Optional[MCTSBackend] = None,
    config: Optional[MCTSConfig] = None,
    use_batched: bool = True
) -> MCTSBase:
    """Factory function to create MCTS with specified backend.

    Args:
        backend: Which MCTS implementation to use (overrides config.backend if provided)
        config: MCTS configuration (includes backend setting)
        use_batched: For C++ backend, use CppBatchedMCTS (proper AlphaZero) if True,
                     or CppMCTS (fast but root-only) if False. Default: True.

    Returns:
        MCTS instance

    Backend selection priority:
        1. Explicit `backend` parameter (if provided)
        2. `config.backend` (always respected when config is provided)
        3. Auto-detect best available backend (only when no config provided)

    Note on C++ backends:
        - CppBatchedMCTS (use_batched=True): Proper AlphaZero where every leaf
          gets a neural network evaluation. Slower but produces high-quality
          training data. RECOMMENDED FOR TRAINING.
        - CppMCTS (use_batched=False): Fast but only evaluates root node.
          Leaf expansions use uniform priors. Use for inference/play only.
    """
    if config is None:
        config = MCTSConfig()
        # No config provided - auto-detect best backend
        if backend is None:
            backend = get_best_backend()
    else:
        # Config provided - use config.backend unless explicitly overridden
        if backend is None:
            backend = config.backend

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
            if use_batched:
                # Proper AlphaZero with batch leaf evaluation (recommended for training)
                from .cpp import CppBatchedMCTS
                return CppBatchedMCTS(config)
            else:
                # Fast but root-only evaluation (for inference/play)
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
