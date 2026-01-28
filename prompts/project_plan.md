# AlphaZero Chess Engine Implementation Plan

## Overview

Build an AlphaZero chess engine from scratch using PyTorch, mapping directly to the paper's mathematical concepts while optimizing for single-node (CPU + NVIDIA GPU) training.

## Project Structure

```
alpha-zero-chess/
├── alphazero/
│   ├── __init__.py
│   ├── config.py                 # Global configuration dataclasses
│   │
│   ├── chess_env/                # Chess game interface
│   │   ├── __init__.py
│   │   ├── board.py              # GameState wrapper around python-chess
│   │   ├── encoding.py           # Board → tensor encoding (119 planes)
│   │   └── moves.py              # Move ↔ action index mapping (4672 actions)
│   │
│   ├── neural/                   # Neural network
│   │   ├── __init__.py
│   │   ├── network.py            # AlphaZeroNetwork (ResNet + heads)
│   │   ├── blocks.py             # ResidualBlock, ConvBlock
│   │   └── loss.py               # Policy + Value loss functions
│   │
│   ├── mcts/                     # Monte Carlo Tree Search (multi-backend)
│   │   ├── __init__.py           # Backend selection factory
│   │   ├── base.py               # Abstract base classes (MCTSNode, MCTS)
│   │   ├── evaluator.py          # Neural network evaluator interface
│   │   │
│   │   ├── python/               # Pure Python implementation (educational)
│   │   │   ├── __init__.py
│   │   │   ├── node.py           # MCTSNode with PUCT
│   │   │   └── search.py         # MCTS search algorithm
│   │   │
│   │   ├── cython/               # Cython-optimized implementation
│   │   │   ├── __init__.py
│   │   │   ├── node.pyx          # Cython MCTSNode
│   │   │   ├── search.pyx        # Cython MCTS search
│   │   │   └── setup.py          # Cython build configuration
│   │   │
│   │   └── cpp/                  # C++ implementation (pybind11)
│   │       ├── __init__.py       # Python bindings loader
│   │       ├── src/
│   │       │   ├── node.hpp      # C++ MCTSNode
│   │       │   ├── node.cpp
│   │       │   ├── search.hpp    # C++ MCTS search
│   │       │   ├── search.cpp
│   │       │   └── bindings.cpp  # pybind11 bindings
│   │       └── CMakeLists.txt    # CMake build configuration
│   │
│   ├── training/                 # Training pipeline
│   │   ├── __init__.py
│   │   ├── trajectory.py         # TrajectoryState, Trajectory dataclasses
│   │   ├── replay_buffer.py      # Fixed-size replay buffer
│   │   └── learner.py            # Training loop with SGD
│   │
│   ├── selfplay/                 # Self-play generation
│   │   ├── __init__.py
│   │   ├── actor.py              # Self-play actor (runs in subprocess)
│   │   ├── game.py               # Single game execution
│   │   └── coordinator.py        # Multi-process orchestration
│   │
│   └── evaluation/               # Strength evaluation
│       ├── __init__.py
│       ├── arena.py              # Match play between agents
│       └── stockfish.py          # Stockfish integration for Elo estimation
│
├── demo/                         # Educational demo scripts
│   ├── mcts_demo.py              # Step-by-step MCTS visualization
│   ├── network_demo.py           # Neural network forward pass demo
│   └── selfplay_demo.py          # Single game self-play walkthrough
│
├── scripts/
│   ├── train.py                  # Main training entry point
│   ├── play.py                   # Play against trained model
│   ├── evaluate.py               # Evaluate model strength
│   └── benchmark_mcts.py         # Compare MCTS backend performance
│
├── tests/                        # Unit tests
├── checkpoints/                  # Saved models
└── logs/                         # Training logs
```

## Core Components

### 1. Chess Environment (`chess_env/`)

**Board Representation (119 planes, 8×8):**
- Planes 0-11: Current position (6 piece types × 2 colors)
- Planes 12-23: Position T-1 (one move ago)
- ... (8 history positions total = 96 planes)
- Planes 96-99: Castling rights (4 planes)
- Plane 100: Side to move
- Planes 101-108: Repetition counters
- Planes 109-118: Move clocks (no-progress count, total moves)

**Move Encoding (4672 actions):**
- Queen-like moves from each square: 56 directions × 64 squares = 3584
- Knight moves: 8 × 64 = 512
- Underpromotions: 9 × 64 = 576 (3 piece types × 3 directions × 64)

```python
# Key classes
class GameState:
    """Immutable game state wrapper around python-chess Board."""
    def get_observation(self) -> np.ndarray  # (119, 8, 8)
    def get_legal_actions(self) -> np.ndarray  # (4672,) binary mask
    def apply_action(self, action: int) -> GameState
    def is_terminal(self) -> bool
    def get_result(self) -> float  # 1.0 win, -1.0 loss, 0.0 draw

class MoveEncoder:
    """Bidirectional mapping between chess.Move and action indices."""
    def encode(self, move: chess.Move) -> int
    def decode(self, action: int, board: chess.Board) -> chess.Move
```

### 2. Neural Network (`neural/`)

**Architecture (matching AlphaZero paper):**
- Input: 119 × 8 × 8 tensor
- Initial conv: 3×3, 192 filters → BatchNorm → ReLU
- Residual tower: 15 residual blocks (each: 2× [3×3 conv → BN → ReLU] + skip)
- Policy head: 1×1 conv (2 filters) → BN → ReLU → flatten → FC(4672) → mask → log_softmax
- Value head: 1×1 conv (1 filter) → BN → ReLU → flatten → FC(192) → ReLU → FC(1) → tanh

```python
class AlphaZeroNetwork(nn.Module):
    def __init__(self, input_channels=119, num_filters=192, num_blocks=15, num_actions=4672):
        ...

    def forward(self, x: Tensor, legal_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (policy_log_probs, value)."""
        ...
```

**Loss Function:**
```
L = L_policy + L_value
L_policy = -π^T · log(p)           # Cross-entropy with MCTS policy
L_value = (z - v)²                  # MSE with game outcome
```

### 3. MCTS (`mcts/`) - Multi-Backend Architecture

**PUCT Selection (from paper):**
```
a* = argmax_a [ Q(s,a) + c_puct · P(s,a) · √(N(s)) / (1 + N(s,a)) ]

Where:
- Q(s,a) = mean value of action a
- P(s,a) = prior probability from neural network
- N(s) = visit count of parent
- N(s,a) = visit count of action a
- c_puct = 1.25 (exploration constant)
```

**Dirichlet Noise at Root:**
```
P(s,a) = (1 - ε) · p_a + ε · η_a
Where η ~ Dir(α), α = 0.3 for chess, ε = 0.25
```

**Temperature-based Action Selection:**
```
π(a) ∝ N(s,a)^(1/τ)
τ = 1.0 for first 30 moves, τ → 0 (greedy) thereafter
```

**Backend Selection:**
```python
# mcts/__init__.py - Factory for backend selection
from enum import Enum

class MCTSBackend(Enum):
    PYTHON = "python"    # Pure Python (educational, readable)
    CYTHON = "cython"    # Cython-optimized (~5-10x faster)
    CPP = "cpp"          # C++ with pybind11 (~20-50x faster)

def create_mcts(backend: MCTSBackend = MCTSBackend.PYTHON, config: MCTSConfig = None) -> MCTS:
    """Factory function to create MCTS with specified backend."""
    if backend == MCTSBackend.PYTHON:
        from .python.search import PythonMCTS
        return PythonMCTS(config)
    elif backend == MCTSBackend.CYTHON:
        from .cython.search import CythonMCTS
        return CythonMCTS(config)
    elif backend == MCTSBackend.CPP:
        from .cpp import CppMCTS
        return CppMCTS(config)
```

**Abstract Base Classes (`mcts/base.py`):**
```python
from abc import ABC, abstractmethod

@dataclass
class MCTSConfig:
    num_simulations: int = 800
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30

class MCTSNodeBase(ABC):
    """Abstract base for MCTS nodes - all backends implement this interface."""
    prior: float           # P(s,a) from neural network
    visit_count: int       # N(s,a)
    value_sum: float       # W(s,a) = sum of values

    @property
    @abstractmethod
    def q_value(self) -> float: ...

    @abstractmethod
    def expand(self, priors: np.ndarray, legal_mask: np.ndarray) -> None: ...

    @abstractmethod
    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNodeBase']: ...

class MCTSBase(ABC):
    """Abstract base for MCTS search - all backends implement this interface."""

    @abstractmethod
    def search(self, state: GameState, move_number: int) -> Tuple[np.ndarray, MCTSNodeBase]:
        """Run MCTS and return (policy, root_node)."""
        ...
```

**Pure Python Implementation (`mcts/python/`):**
- Fully documented with comments mapping to paper equations
- Uses NumPy for vectorized operations where possible
- Ideal for understanding and debugging

**Cython Implementation (`mcts/cython/`):**
- Typed memoryviews for array operations
- cdef classes for nodes (no Python overhead)
- Inline PUCT calculation
- Expected speedup: 5-10x over pure Python

**C++ Implementation (`mcts/cpp/`):**
- Header-only node implementation for inlining
- Custom memory pool for node allocation
- SIMD-optimized PUCT calculation (optional)
- pybind11 bindings with numpy array support
- Expected speedup: 20-50x over pure Python

**Benchmark Script (`scripts/benchmark_mcts.py`):**
```python
def benchmark_mcts_backends():
    """Compare performance of all MCTS backends."""
    # Metrics:
    # - Simulations per second
    # - Memory usage
    # - Time to complete 100 searches (800 sims each)
    # - Scaling with simulation count

    results = {}
    for backend in [MCTSBackend.PYTHON, MCTSBackend.CYTHON, MCTSBackend.CPP]:
        mcts = create_mcts(backend, config)
        # Run benchmarks...
        results[backend] = {
            'sims_per_sec': ...,
            'memory_mb': ...,
            'time_100_searches': ...,
        }

    # Print comparison table and generate plots
```

### 4. Training Pipeline (`training/`)

**Trajectory Storage:**
```python
@dataclass
class TrajectoryState:
    observation: np.ndarray   # (119, 8, 8)
    legal_mask: np.ndarray    # (4672,)
    policy: np.ndarray        # (4672,) MCTS visit distribution
    value: float              # Game outcome from this player's view
    action: int               # Action taken
    player: int               # 0=white, 1=black

@dataclass
class Trajectory:
    states: List[TrajectoryState]
    result: float             # 1.0 white wins, -1.0 black wins, 0.0 draw
```

**Replay Buffer:**
- Fixed size: 1,000,000 positions (most recent)
- Uniform random sampling
- Thread-safe for concurrent actor writes

**Learner:**
- Optimizer: SGD with momentum (lr=0.2, momentum=0.9, weight_decay=1e-4)
- LR schedule: drops at 100k, 300k, 500k steps (×0.1 each)
- Batch size: 4096
- Mixed precision (AMP) for GPU efficiency
- Gradient clipping: max_norm=1.0

### 5. Self-Play Pipeline (`selfplay/`)

**Parallelization Strategy:**
```
┌─────────────────────────────────────────────────────────────┐
│                      Main Process                            │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │   Learner   │◄───│ ReplayBuffer │◄───│  Trajectory   │  │
│  │   (GPU)     │    │              │    │    Queue      │  │
│  └──────┬──────┘    └──────────────┘    └───────▲───────┘  │
│         │                                        │          │
│         │ Weight updates (periodic)              │          │
│         ▼                                        │          │
│  ┌──────────────────────────────────────────────┴───────┐  │
│  │              Weight Distribution Queue                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ Actor 0 │   │ Actor 1 │   │ Actor 2 │   │ Actor N │
    │  (CPU)  │   │  (CPU)  │   │  (CPU)  │   │  (CPU)  │
    └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

- **Actors**: Run in separate processes (multiprocessing)
- **GPU**: Dedicated to learner for training; actors use CPU for MCTS
- **Communication**: Queues for trajectories (actor→learner) and weights (learner→actors)
- **Batched inference**: Optional GPU inference server for actors (advanced)

### 6. Evaluation (`evaluation/`)

- **Self-play evaluation**: New model vs previous checkpoint
- **Stockfish evaluation**: Play matches against Stockfish at various levels
- **Elo estimation**: Based on win/draw/loss rates

## Key Hyperparameters (AlphaZero Paper)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Residual blocks | 15 | Medium size (can scale to 19 later) |
| Filters | 192 | Medium size (can scale to 256 later) |
| MCTS simulations | 800 | Per move during self-play |
| c_puct | 1.25 | Exploration constant |
| Dirichlet α | 0.3 | Chess-specific (0.03 for Go) |
| Dirichlet ε | 0.25 | Noise weight at root |
| Temperature | 1.0 → 0 | τ=1 for moves 1-30, then greedy |
| Replay buffer | 1M positions | ~500k games |
| Batch size | 4096 | Training batch |
| Learning rate | 0.2 → 0.002 | Drops at 100k, 300k, 500k |
| Weight decay | 1e-4 | L2 regularization |

## Implementation Order

1. **Phase 1: Chess Environment**
   - `chess_env/encoding.py` - Board to tensor (119 planes)
   - `chess_env/moves.py` - Move encoding (4672 actions)
   - `chess_env/board.py` - GameState wrapper

2. **Phase 2: Neural Network**
   - `neural/blocks.py` - ResidualBlock
   - `neural/network.py` - AlphaZeroNetwork (15 blocks, 192 filters)
   - `neural/loss.py` - Loss functions

3. **Phase 3: MCTS (Pure Python First)**
   - `mcts/base.py` - Abstract base classes
   - `mcts/evaluator.py` - Neural evaluator interface
   - `mcts/python/node.py` - Pure Python MCTSNode
   - `mcts/python/search.py` - Pure Python MCTS algorithm

4. **Phase 4: Demo Scripts**
   - `demo/mcts_demo.py` - Step-by-step MCTS visualization
   - `demo/network_demo.py` - Neural network walkthrough
   - `demo/selfplay_demo.py` - Single game demonstration

5. **Phase 5: Training Infrastructure**
   - `training/trajectory.py` - Data structures
   - `training/replay_buffer.py` - Buffer
   - `training/learner.py` - Training loop

6. **Phase 6: Self-Play Pipeline**
   - `selfplay/game.py` - Single game
   - `selfplay/actor.py` - Actor process (CPU inference default)
   - `selfplay/coordinator.py` - Multi-process orchestration

7. **Phase 7: Optimized MCTS Backends**
   - `mcts/cython/` - Cython implementation
   - `mcts/cpp/` - C++ implementation with pybind11
   - `mcts/__init__.py` - Backend factory

8. **Phase 8: Evaluation & Benchmarks**
   - `evaluation/` - Stockfish integration
   - `scripts/benchmark_mcts.py` - Backend performance comparison
   - `scripts/train.py` - Main entry point
   - `scripts/play.py` - Interactive play
   - `scripts/evaluate.py` - Elo estimation

## Verification Strategy

1. **Unit Tests:**
   - Move encoding roundtrip (encode → decode)
   - Board encoding correctness
   - MCTS node statistics
   - Network forward pass shapes

2. **Integration Tests:**
   - Single self-play game completion
   - Training step execution
   - Checkpoint save/load

3. **Functional Tests:**
   - Train on small network (5 blocks, 64 filters) for a few hundred games
   - Verify loss decreases
   - Verify model beats random play

4. **Performance Benchmarks:**
   - MCTS simulations per second
   - Training throughput (positions/second)
   - Self-play games per hour

## Scalability Notes

To reach Grandmaster level (Elo 2500+):
- Full 19-block, 256-filter network
- ~700,000 training steps
- ~44 million self-play games (AlphaZero paper)
- Estimated training time on single GPU: weeks to months
- Consider: gradient checkpointing, larger batch with accumulation, distributed training later
