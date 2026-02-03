# AlphaZero Chess - Project Structure Documentation

**Generated:** 2026-02-02
**Project:** AlphaZero Chess Engine with C++ MCTS Backend

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Core Modules](#core-modules)
4. [Scripts Documentation](#scripts-documentation)
5. [C++ Extension](#c-extension)
6. [Dependencies Map](#dependencies-map)

---

## Project Overview

This is a complete AlphaZero implementation for chess featuring:
- Pure Python, Cython, and C++ MCTS backends
- PyTorch neural network (ResNet architecture)
- Self-play training pipeline with replay buffer
- Evaluation against Stockfish
- Web interface for playing against the AI

**Key Performance:**
- C++ MCTS: 7,280 simulations/sec with neural network (RTX 4060)
- Move generation: 189-422M nps
- Batch inference: 24,000 positions/sec (batch=64)

---

## Directory Structure

```
alpha-zero-chess/
├── alphazero/              # Main Python package
│   ├── scripts/            # Python training/evaluation scripts
│   │   ├── train.py        # Main training script
│   │   ├── evaluate.py     # Model evaluation
│   │   ├── play.py         # Interactive play
│   │   └── benchmark_mcts.py # MCTS benchmarks
│   ├── chess_env/          # Chess game state and encoding
│   ├── mcts/               # MCTS implementations (Python/Cython only, C++ removed)
│   ├── neural/             # Neural network architecture
│   ├── selfplay/           # Self-play game generation
│   ├── training/           # Training loop and learner
│   ├── evaluation/         # Model evaluation and arena (backend-agnostic)
│   └── config.py           # Configuration dataclasses
│
├── alphazero-cpp/          # C++ extension (pybind11)
│   ├── scripts/            # C++ backend training scripts
│   │   └── train.py        # C++ MCTS training script
│   ├── include/            # C++ headers
│   ├── src/                # C++ source files
│   └── tests/              # C++ benchmarks
│
├── tests/                  # Python tests
├── checkpoints/            # Saved model checkpoints
├── docs/                   # Documentation
└── web/                    # Web interface
```

---

## Core Modules

### 1. `alphazero/` - Main Python Package

#### `alphazero/chess_env/` - Chess Environment
**Purpose:** Chess game state representation and encoding

**Files:**
- `game_state.py` - Chess game state wrapper around python-chess
- `observation.py` - Board encoding to 119-plane representation
- `moves.py` - Move encoding to 4672-action space
- `moves_cpp_aligned.py` - Python move encoder aligned with C++ implementation

**Key Features:**
- 119-plane board representation (piece positions, castling, en passant, etc.)
- 4672-action space (all possible chess moves)
- Symmetry augmentation for training data

---

#### `alphazero/mcts/` - Monte Carlo Tree Search
**Purpose:** MCTS implementations with multiple backends

**Structure:**
```
mcts/
├── base.py              # Abstract base classes
├── evaluator.py         # Neural network evaluator interface
├── python/              # Pure Python MCTS
│   ├── search.py
│   ├── node.py
│   └── parallel.py      # Parallel MCTS with virtual loss
└── cython/              # Cython-optimized MCTS (5-10x faster)
    └── search.pyx

# Note: C++ MCTS backend has been moved to alphazero-cpp package
```

**Backends:**
1. **Python** (`python/search.py`)
   - Educational, readable implementation
   - ~1K simulations/sec
   - Used for debugging and understanding

2. **Cython** (`cython/search.pyx`)
   - 5-10x faster than Python
   - Requires compilation: `python setup.py build_ext --inplace`

**Note:** C++ MCTS backend has been moved to the `alphazero-cpp` package for better organization. Use `import alphazero_cpp` to access C++ MCTS directly.

**Key Classes:**
- `MCTSBase` - Abstract base class for all MCTS implementations
- `NetworkEvaluator` - Wraps neural network for position evaluation
- `create_mcts()` - Factory function to create MCTS with specified backend

---

#### `alphazero/neural/` - Neural Network
**Purpose:** AlphaZero neural network architecture

**Files:**
- `network.py` - Main AlphaZeroNetwork class
- `blocks.py` - ResNet building blocks (ConvBlock, ResBlock, PolicyHead, ValueHead)
- `loss.py` - AlphaZero loss function (policy + value)

**Architecture:**
```
Input (119, 8, 8)
    ↓
ConvBlock (119 → filters)
    ↓
ResBlock × num_blocks
    ↓
├─→ PolicyHead → (4672,) softmax
└─→ ValueHead → (1,) tanh
```

**Default Sizes:**
- Small: 64 filters × 5 blocks (~1M params)
- Medium: 128 filters × 10 blocks (~5M params)
- Large: 192 filters × 15 blocks (~20M params) ← Default for cpptrain.py

---

#### `alphazero/selfplay/` - Self-Play
**Purpose:** Generate training data through self-play

**Files:**
- `game.py` - Single self-play game execution
- `actor.py` - Self-play actor (single process)
- `batched_actor.py` - Batched actor with inference server
- `coordinator.py` - Coordinates multiple actors
- `inference_server.py` - Centralized GPU inference server

**Architecture:**
```
Coordinator
    ↓
├─→ Actor 1 ──┐
├─→ Actor 2 ──┼─→ Inference Server (GPU)
├─→ Actor 3 ──┤       ↓
└─→ Actor 4 ──┘   Neural Network
    ↓
Replay Buffer
```

**Key Features:**
- Multi-process self-play with centralized GPU inference
- Batched inference for efficiency
- Replay buffer with experience replay
- Temperature-based move selection

---

#### `alphazero/training/` - Training
**Purpose:** Neural network training loop

**Files:**
- `learner.py` - Training loop with optimizer
- `replay_buffer.py` - Experience replay buffer
- `trajectory.py` - Game trajectory storage

**Training Loop:**
1. Sample batch from replay buffer
2. Forward pass through network
3. Compute loss (policy cross-entropy + value MSE)
4. Backward pass and optimizer step
5. Broadcast updated weights to actors

---

#### `alphazero/evaluation/` - Evaluation
**Purpose:** Evaluate model strength

**Files:**
- `arena.py` - Play matches between two models
- `stockfish.py` - Stockfish integration
- `elo.py` - ELO rating calculation

**Features:**
- Head-to-head matches between models
- Stockfish evaluation at different skill levels
- ELO rating tracking

---

#### `alphazero/config.py` - Configuration
**Purpose:** Centralized configuration with dataclasses

**Key Configs:**
- `MCTSConfig` - MCTS parameters (simulations, c_puct, etc.)
- `NetworkConfig` - Network architecture
- `TrainingConfig` - Training hyperparameters
- `SelfPlayConfig` - Self-play settings
- `AlphaZeroConfig` - Complete configuration

**Profiles:**
- `PROFILES['high']` - A100/H100 GPUs
- `PROFILES['mid']` - T4/V100 GPUs
- `PROFILES['low']` - RTX 4060 GPUs

---

## Scripts Documentation

### `alphazero/scripts/train.py` - Main Training Script

**Purpose:** Full-featured AlphaZero training with multi-process self-play

**Imports:**
```python
# From alphazero package
from alphazero import (
    AlphaZeroConfig, MCTSConfig, NetworkConfig,
    TrainingConfig, ReplayBufferConfig, MCTSBackend,
    PROFILES, TrainingProfile
)
from alphazero.neural import AlphaZeroNetwork, count_parameters
from alphazero.selfplay import (
    SelfPlayCoordinator, BatchedSelfPlayCoordinator
)
from alphazero.mcts import get_available_backends, get_best_backend
```

**What It Does:**
1. Creates neural network (AlphaZeroNetwork)
2. Spawns multiple self-play actors (ActorProcess)
3. Starts centralized inference server (InferenceServer)
4. Actors generate games using MCTS + neural network
5. Games stored in replay buffer
6. Learner trains on batches from replay buffer
7. Updated weights broadcast to actors
8. Repeat for N iterations

**Key Features:**
- Multi-process architecture with spawn method (CUDA-safe)
- Centralized GPU inference server for efficiency
- Batched inference (512 positions/batch)
- Hardware profiles (high/mid/low)
- Iterative training with buffer refresh
- Checkpoint saving and resuming
- Mixed precision training (AMP)

**Command-Line Arguments:**
```bash
--profile high/mid/low      # Hardware profile
--steps N                   # Total training steps
--iterations N              # Number of iterations
--steps-per-iteration N     # Steps per iteration
--actors N                  # Number of self-play actors
--batch-size N              # Training batch size
--min-buffer N              # Min buffer size before training
--filters N                 # Network filters
--blocks N                  # Residual blocks
--simulations N             # MCTS simulations per move
--mcts-backend python/cython/cpp/auto
--batched-inference         # Use centralized inference server
--inference-batch-size N    # Inference batch size
--inference-timeout F       # Inference timeout (seconds)
--checkpoint-dir PATH       # Checkpoint directory
--resume PATH               # Resume from checkpoint
--device cuda/cpu           # Device
--no-amp-training           # Disable mixed precision training
--no-amp-inference          # Disable mixed precision inference
--verbose                   # Verbose logging
```

**Example Usage:**
```bash
# Low-end GPU (RTX 4060)
uv run python scripts/train.py --profile low

# Custom settings
uv run python scripts/train.py \
    --iterations 100 \
    --steps-per-iteration 1000 \
    --actors 4 \
    --simulations 800 \
    --mcts-backend cpp

# Resume training
uv run python scripts/train.py --resume checkpoints/checkpoint_50.pt
```

**Dependencies:**
- `alphazero.neural.AlphaZeroNetwork` - Neural network
- `alphazero.selfplay.BatchedSelfPlayCoordinator` - Coordinates actors
- `alphazero.selfplay.InferenceServer` - GPU inference
- `alphazero.mcts.create_mcts` - MCTS factory
- `alphazero.training.Learner` - Training loop

---

### `alphazero-cpp/scripts/train.py` - C++ MCTS Training Script

**Purpose:** Simplified training script focused on C++ MCTS backend

**Imports:**
```python
# C++ extension (direct import)
import alphazero_cpp

# Standard libraries only (no alphazero package imports)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import chess
```

**What It Does:**
1. Creates 192×15 AlphaZero network (self-contained definition)
2. Single-process self-play using C++ BatchedMCTS
3. Batched GPU inference during MCTS search
4. Stores games in replay buffer
5. Trains network on sampled batches
6. Saves checkpoints with "cpp" prefix

**Key Features:**
- **Self-contained:** Defines network in the script (no alphazero package dependency)
- **C++ MCTS only:** Uses `alphazero_cpp.BatchedMCTSSearch` directly
- **Batched leaf evaluation:** Proper AlphaZero with NN eval at every leaf
- **Performance metrics:** Detailed tracking of self-play and training performance
- **Simple architecture:** Single process, easier to understand and debug
- **192×15 network:** ~20M parameters (production size)

**Command-Line Arguments:**
```bash
--iterations N              # Training iterations (default: 100)
--games-per-iter N          # Games per iteration (default: 50)
--simulations N             # MCTS sims per move (default: 800)
--batch-size N              # Training batch size (default: 256)
--mcts-batch-size N         # MCTS leaf batch size (default: 64)
--lr F                      # Learning rate (default: 0.001)
--filters N                 # Network filters (default: 192)
--blocks N                  # Residual blocks (default: 15)
--buffer-size N             # Replay buffer size (default: 100000)
--epochs N                  # Training epochs/iter (default: 5)
--temperature-moves N       # Temp=1 moves (default: 30)
--c-puct F                  # MCTS exploration (default: 1.5)
--device cuda/cpu           # Device (default: cuda)
--save-dir PATH             # Checkpoint dir (default: checkpoints)
--resume PATH               # Resume from checkpoint
--save-interval N           # Save every N iters (default: 5)
```

**Example Usage:**
```bash
# Basic training
uv run python scripts/cpptrain.py

# Fast iteration
uv run python scripts/cpptrain.py \
    --iterations 200 \
    --games-per-iter 25 \
    --simulations 400

# High quality
uv run python scripts/cpptrain.py \
    --simulations 1600 \
    --games-per-iter 100

# Resume
uv run python scripts/cpptrain.py --resume checkpoints/cpp_iter_50.pt
```

**Performance Metrics Tracked:**
- Self-play: moves/sec, sims/sec, NN evals/sec
- Training: loss, policy loss, value loss, samples/sec
- Games: win/draw/loss distribution, avg game length
- Buffer: size, positions added
- Timing: iteration time, ETA

**Dependencies:**
- `alphazero_cpp` - C++ extension (BatchedMCTSSearch, encode_position, move_to_index)
- `torch` - PyTorch for neural network
- `chess` - python-chess for board state

**Checkpoints:**
- Saved as `checkpoints/cpp_iter_N.pt`
- Contains: model_state_dict, optimizer_state_dict, iteration, config

---

### `alphazero/scripts/evaluate.py` - Model Evaluation

**Purpose:** Evaluate trained models against each other or Stockfish

**Imports:**
```python
from alphazero import AlphaZeroConfig, MCTSConfig
from alphazero.neural import AlphaZeroNetwork
from alphazero.evaluation import Arena, StockfishPlayer
from alphazero.mcts import create_mcts
from alphazero.mcts.evaluator import NetworkEvaluator
```

**What It Does:**
1. Loads two model checkpoints (or one model vs Stockfish)
2. Creates MCTS players for each model
3. Plays N games between them
4. Reports win/draw/loss statistics
5. Calculates ELO rating difference

**Command-Line Arguments:**
```bash
--model1 PATH               # First model checkpoint
--model2 PATH               # Second model checkpoint (or "stockfish")
--games N                   # Number of games to play
--simulations N             # MCTS simulations per move
--stockfish-skill N         # Stockfish skill level (0-20)
--device cuda/cpu           # Device
--verbose                   # Show game details
```

**Example Usage:**
```bash
# Compare two models
uv run python scripts/evaluate.py \
    --model1 checkpoints/cpp_iter_50.pt \
    --model2 checkpoints/cpp_iter_100.pt \
    --games 100

# Evaluate against Stockfish
uv run python scripts/evaluate.py \
    --model1 checkpoints/cpp_iter_100.pt \
    --model2 stockfish \
    --stockfish-skill 10 \
    --games 50
```

**Dependencies:**
- `alphazero.neural.AlphaZeroNetwork` - Load models
- `alphazero.evaluation.Arena` - Play matches
- `alphazero.evaluation.StockfishPlayer` - Stockfish integration
- `alphazero.mcts.create_mcts` - Create MCTS players

---

### `alphazero/scripts/benchmark_mcts.py` - MCTS Backend Benchmark

**Purpose:** Compare performance of different MCTS backends

**Imports:**
```python
from alphazero import MCTSConfig, MCTSBackend
from alphazero.chess_env import GameState
from alphazero.mcts import create_mcts, get_available_backends
from alphazero.mcts.evaluator import RandomEvaluator
```

**What It Does:**
1. Tests Python, Cython, and C++ MCTS backends
2. Runs N searches with M simulations each
3. Measures time per search and simulations/sec
4. Compares speedup vs Python baseline

**Command-Line Arguments:**
```bash
--searches N                # Number of searches (default: 100)
--simulations N             # Simulations per search (default: 800)
--backends LIST             # Backends to test (default: all)
```

**Example Usage:**
```bash
# Benchmark all backends
uv run python scripts/benchmark_mcts.py

# Quick test
uv run python scripts/benchmark_mcts.py --searches 10 --simulations 100

# Compare specific backends
uv run python scripts/benchmark_mcts.py --backends python cpp
```

**Note:** This uses `RandomEvaluator` (instant), so it measures tree operations only, not neural network inference.

**Dependencies:**
- `alphazero.mcts.create_mcts` - Create MCTS instances
- `alphazero.mcts.evaluator.RandomEvaluator` - Fast dummy evaluator
- `alphazero.chess_env.GameState` - Chess state

---

## C++ Extension

### `alphazero-cpp/` - C++ MCTS Backend

**Purpose:** High-performance MCTS implementation with pybind11 bindings

**Structure:**
```
alphazero-cpp/
├── include/
│   ├── chess/              # Chess move generation
│   │   ├── board.hpp
│   │   └── movegen.hpp
│   ├── encoding/           # Position and move encoding
│   │   ├── position_encoder.hpp
│   │   └── move_encoder.hpp
│   └── mcts/               # MCTS search
│       ├── node.hpp
│       ├── search.hpp
│       └── batch_search.hpp
│
├── src/
│   ├── chess/              # Chess implementation
│   ├── encoding/           # Encoding implementation
│   ├── mcts/               # MCTS implementation
│   │   ├── node.cpp
│   │   ├── search.cpp      # CppMCTS (root-only eval)
│   │   └── batch_search.cpp # CppBatchedMCTS (proper AlphaZero)
│   └── bindings/
│       └── python_bindings.cpp  # pybind11 bindings
│
├── tests/                  # C++ benchmarks
│   ├── benchmark_with_nn.py
│   ├── benchmark_train_path.py
│   └── test_phase4_integration.py
│
├── train.py                # Standalone training script
└── CMakeLists.txt          # Build configuration
```

**Key Components:**

#### 1. Chess Engine (`chess/`)
- Bitboard-based move generation
- Performance: 189-422M nps
- Based on chess-library (third-party)

#### 2. Encoding (`encoding/`)
- `PositionEncoder`: Board → 119 planes (NHWC format)
- `MoveEncoder`: Move ↔ 4672 action index
- Aligned with Python implementation

#### 3. MCTS (`mcts/`)

**`search.cpp` - CppMCTS (Fast but simplified)**
- Only evaluates root node with neural network
- Leaf expansions use uniform priors
- Fast: 279K sims/sec (no NN bottleneck)
- Use for: Inference, playing games
- **Not recommended for training** (lower quality data)

**`batch_search.cpp` - CppBatchedMCTS (Proper AlphaZero)**
- Evaluates every leaf node with neural network
- Batched leaf collection for GPU efficiency
- Proper: 7,280 sims/sec (with NN on RTX 4060)
- Use for: Training
- **Recommended for training** (high quality data)

**Key Methods:**
```cpp
// CppBatchedMCTS
void init_search(fen, root_policy, root_value)
tuple<int, array, array> collect_leaves()  // Returns (num_leaves, obs_batch, mask_batch)
void update_leaves(policies, values)
bool is_complete()
vector<int> get_visit_counts()
void reset()
```

#### 4. Python Bindings (`bindings/python_bindings.cpp`)
Exposes to Python:
- `MCTSSearch` - CppMCTS class
- `BatchedMCTSSearch` - CppBatchedMCTS class
- `encode_position(fen)` - Position encoding
- `move_to_index(uci, fen)` - Move to action index
- `index_to_move(idx, fen)` - Action index to move

**Build Instructions:**
```bash
cd alphazero-cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**Output:** `alphazero-cpp/build/Release/alphazero_cpp.pyd` (Windows) or `.so` (Linux)

---

## Dependencies Map

### Script Dependencies

```
scripts/train.py
├── alphazero.neural.AlphaZeroNetwork
├── alphazero.selfplay.BatchedSelfPlayCoordinator
│   ├── alphazero.selfplay.BatchedActorProcess
│   │   ├── alphazero.selfplay.BatchedActor
│   │   │   ├── alphazero.mcts.create_mcts
│   │   │   │   └── alphazero.mcts.cpp.CppBatchedMCTS
│   │   │   └── alphazero.chess_env.GameState
│   │   └── alphazero.selfplay.game.SelfPlayGame
│   └── alphazero.selfplay.InferenceServer
│       └── alphazero.selfplay.BatchedEvaluator
├── alphazero.training.Learner
│   └── alphazero.training.ReplayBuffer
└── alphazero.config (all config classes)

scripts/cpptrain.py
├── alphazero_cpp (C++ extension)
│   ├── BatchedMCTSSearch
│   ├── encode_position
│   ├── move_to_index
│   └── index_to_move
├── torch (PyTorch)
└── chess (python-chess)

scripts/evaluate.py
├── alphazero.neural.AlphaZeroNetwork
├── alphazero.evaluation.Arena
├── alphazero.evaluation.StockfishPlayer
├── alphazero.mcts.create_mcts
└── alphazero.mcts.evaluator.NetworkEvaluator

scripts/benchmark_mcts.py
├── alphazero.mcts.create_mcts
├── alphazero.mcts.evaluator.RandomEvaluator
└── alphazero.chess_env.GameState
```

### Module Dependencies

```
alphazero/
├── config.py (no internal deps)
├── chess_env/
│   └── (python-chess only)
├── mcts/
│   ├── base.py (no deps)
│   ├── evaluator.py → neural/
│   ├── python/ → base, chess_env
│   ├── cython/ → base, chess_env
│   └── cpp/ → alphazero_cpp (C++ extension)
├── neural/
│   └── (torch only)
├── selfplay/
│   ├── game.py → chess_env, mcts
│   ├── actor.py → chess_env, mcts, neural
│   ├── batched_actor.py → chess_env, mcts, neural
│   ├── coordinator.py → selfplay.actor, training
│   └── inference_server.py → neural
├── training/
│   ├── learner.py → neural, training.replay_buffer
│   ├── replay_buffer.py (no deps)
│   └── trajectory.py (no deps)
└── evaluation/
    ├── arena.py → chess_env, mcts
    ├── stockfish.py → chess_env
    └── elo.py (no deps)
```

---

## Key Files Reference

### Configuration
- `alphazero/config.py` - All configuration dataclasses and profiles

### Neural Network
- `alphazero/neural/network.py` - AlphaZeroNetwork class
- `alphazero/neural/blocks.py` - ResNet building blocks
- `alphazero/neural/loss.py` - Loss function

### MCTS
- `alphazero/mcts/__init__.py` - create_mcts() factory
- `alphazero/mcts/base.py` - MCTSBase abstract class
- `alphazero/mcts/cpp/backend.py` - CppBatchedMCTS wrapper
- `alphazero-cpp/src/mcts/batch_search.cpp` - C++ implementation

### Training
- `scripts/train.py` - Full training pipeline
- `scripts/cpptrain.py` - Simplified C++ training
- `alphazero/training/learner.py` - Training loop
- `alphazero/selfplay/coordinator.py` - Self-play coordination

### Evaluation
- `scripts/evaluate.py` - Model evaluation script
- `alphazero/evaluation/arena.py` - Match playing
- `alphazero/evaluation/stockfish.py` - Stockfish integration

---

## Performance Characteristics

### MCTS Backends (with Neural Network on RTX 4060)

| Backend | Sims/sec | Use Case |
|---------|----------|----------|
| Python | ~100 | Debugging, education |
| Cython | ~500 | Medium performance |
| C++ (root-only) | 279K | Fast inference (no NN bottleneck) |
| C++ (batched) | 7,280 | Training (proper AlphaZero) |

### Neural Network Inference (RTX 4060)

| Batch Size | Positions/sec |
|------------|---------------|
| 1 | 372 |
| 8 | 3,115 |
| 32 | 11,529 |
| 64 | 24,047 |
| 128 | 53,277 |
| 256 | 65,564 |

### Training Speed (192×15 network, 800 sims, RTX 4060)

- Self-play: ~5 moves/sec
- Typical game: 50 moves = 10 seconds
- 50 games/iteration = ~8 minutes
- 100 iterations = ~13-14 hours

---

## Common Workflows

### 1. Train a Model
```bash
# Using train.py (multi-process)
uv run python scripts/train.py --profile low --iterations 100

# Using cpptrain.py (single-process, simpler)
uv run python scripts/cpptrain.py --iterations 100 --games-per-iter 50
```

### 2. Evaluate a Model
```bash
# Against another model
uv run python scripts/evaluate.py \
    --model1 checkpoints/cpp_iter_50.pt \
    --model2 checkpoints/cpp_iter_100.pt \
    --games 100

# Against Stockfish
uv run python scripts/evaluate.py \
    --model1 checkpoints/cpp_iter_100.pt \
    --model2 stockfish \
    --stockfish-skill 10
```

### 3. Benchmark MCTS
```bash
# Compare all backends
uv run python scripts/benchmark_mcts.py

# Test C++ backend only
uv run python scripts/benchmark_mcts.py --backends cpp --simulations 800
```

### 4. Build C++ Extension
```bash
cd alphazero-cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

---

## Notes

### C++ MCTS: Two Implementations

**CppMCTS (search.cpp):**
- Fast: 279K sims/sec
- Only evaluates root with NN
- Leaves use uniform priors
- Good for: Playing games, inference
- Bad for: Training (lower quality data)

**CppBatchedMCTS (batch_search.cpp):**
- Proper: 7,280 sims/sec (with NN)
- Evaluates every leaf with NN
- Batched leaf collection
- Good for: Training (high quality data)
- Required for: Proper AlphaZero training

### Training Scripts: Two Options

**train.py:**
- Full-featured multi-process pipeline
- Centralized GPU inference server
- Multiple actors in parallel
- More complex but more efficient
- Uses alphazero package

**cpptrain.py:**
- Simplified single-process
- Self-contained (defines network in script)
- Direct C++ MCTS usage
- Easier to understand and debug
- Detailed performance metrics
- Good for: Learning, experimentation

---

## File Count Summary

```
Total Python files: ~50
Total C++ files: ~30
Total scripts: 4 main scripts
Total tests: ~15
Lines of code: ~15,000 (Python) + ~8,000 (C++)
```

---

**End of Documentation**
