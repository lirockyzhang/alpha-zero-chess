"""Global configuration dataclasses for AlphaZero chess engine."""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class MCTSBackend(Enum):
    """Available MCTS implementation backends."""
    PYTHON = "python"
    CYTHON = "cython"
    CPP = "cpp"


@dataclass
class MCTSConfig:
    """Configuration for Monte Carlo Tree Search."""
    num_simulations: int = 800
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30
    backend: MCTSBackend = MCTSBackend.PYTHON
    batch_size: int = 16  # Number of leaves to batch together during search


@dataclass
class NetworkConfig:
    """Configuration for the neural network architecture."""
    input_channels: int = 122  # Updated to support 8 full historical positions
    num_filters: int = 192
    num_blocks: int = 15
    num_actions: int = 4672
    policy_filters: int = 2
    value_filters: int = 1
    value_hidden: int = 192


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""
    batch_size: int = 4096
    learning_rate: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_schedule_steps: List[int] = field(default_factory=lambda: [100000, 300000, 500000])
    lr_schedule_gamma: float = 0.1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    use_amp_inference: bool = True  # Mixed precision for inference
    checkpoint_interval: int = 1000
    log_interval: int = 100


@dataclass
class ReplayBufferConfig:
    """Configuration for the replay buffer."""
    capacity: int = 1_000_000
    min_size_to_train: int = 10000


@dataclass
class SelfPlayConfig:
    """Configuration for self-play game generation."""
    num_actors: int = 4
    games_per_actor: int = 100
    max_moves: int = 512
    resign_threshold: float = -0.95
    resign_check_moves: int = 5


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    num_games: int = 100
    stockfish_path: Optional[str] = None
    stockfish_elo: int = 1500
    stockfish_time_limit: float = 0.1


@dataclass
class TrainingProfile:
    """Hardware-specific training configuration for optimal performance.

    Profiles are tuned for different GPU memory and compute capabilities:
    - HIGH: A100/H100 with 40-80GB VRAM (cloud training)
    - MID: T4/V100 with 16GB VRAM
    - LOW: RTX 4060/3060 with 8GB VRAM (local development)
    """
    name: str
    filters: int
    blocks: int
    actors: int
    simulations: int
    inference_batch_size: int
    inference_timeout: float
    training_batch_size: int
    replay_buffer_size: int
    min_buffer_size: int
    mcts_backend: str = 'cython'
    mcts_batch_size: int = 16  # Number of leaves to batch together during MCTS


# Hardware profiles for different GPU configurations
PROFILES = {
    'high': TrainingProfile(
        name='high',
        filters=192,
        blocks=15,
        actors=64,
        simulations=800,
        inference_batch_size=1024,
        inference_timeout=0.03,  # 20ms
        training_batch_size=8192,
        replay_buffer_size=1_000_000,
        min_buffer_size=50_000,
        mcts_batch_size=32,  # Larger batches for high-end GPUs
    ),
    'mid': TrainingProfile(
        name='mid',
        filters=192,
        blocks=15,
        actors=32,
        simulations=800,
        inference_batch_size=256,
        inference_timeout=0.015,  # 15ms
        training_batch_size=4096,
        replay_buffer_size=500_000,
        min_buffer_size=20_000,
        mcts_batch_size=16,  # Medium batches for mid-range GPUs
    ),
    'low': TrainingProfile(
        name='low',
        filters=64,
        blocks=5,
        actors=28,
        simulations=800,
        inference_batch_size=128,
        inference_timeout=0.01,  # 10ms
        training_batch_size=2048,
        replay_buffer_size=200_000,
        min_buffer_size=10_000,
        mcts_backend='cpp',
        mcts_batch_size=8,  # Smaller batches for low-end GPUs
    ),
}


@dataclass
class AlphaZeroConfig:
    """Master configuration combining all sub-configs."""
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
