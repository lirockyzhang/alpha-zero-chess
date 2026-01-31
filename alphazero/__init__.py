"""AlphaZero Chess Engine.

A PyTorch implementation of the AlphaZero algorithm for chess.
"""

__version__ = "0.1.0"

from .config import (
    AlphaZeroConfig,
    MCTSConfig,
    NetworkConfig,
    TrainingConfig,
    ReplayBufferConfig,
    SelfPlayConfig,
    EvaluationConfig,
    MCTSBackend,
    TrainingProfile,
    PROFILES,
)
from .utils import parse_checkpoint_architecture, load_checkpoint_with_architecture

__all__ = [
    "AlphaZeroConfig",
    "MCTSConfig",
    "NetworkConfig",
    "TrainingConfig",
    "ReplayBufferConfig",
    "SelfPlayConfig",
    "EvaluationConfig",
    "MCTSBackend",
    "TrainingProfile",
    "PROFILES",
    "parse_checkpoint_architecture",
    "load_checkpoint_with_architecture",
]
