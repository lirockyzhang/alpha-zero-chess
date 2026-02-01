"""C++ MCTS backend module.

This module provides the C++ MCTS backend integration for AlphaZero training.

Two backends are available:
- CppMCTS: Fast but only evaluates root node (not true AlphaZero)
- CppBatchedMCTS: Proper AlphaZero with batch leaf evaluation (recommended)
"""

from .backend import (
    CppMCTS,
    CppBatchedMCTS,
    is_cpp_available,
    create_cpp_mcts,
    CPP_AVAILABLE,
)

__all__ = [
    'CppMCTS',
    'CppBatchedMCTS',
    'is_cpp_available',
    'create_cpp_mcts',
    'CPP_AVAILABLE',
]
