# AlphaZero Chess Engine

A PyTorch implementation of the AlphaZero algorithm for chess, following the methodology from DeepMind's [AlphaZero paper](https://arxiv.org/abs/1712.01815).

## Overview

This project implements a complete AlphaZero chess engine from scratch, including:

- **Neural Network**: ResNet architecture with policy and value heads
- **MCTS**: Monte Carlo Tree Search with PUCT selection
- **Self-Play**: Multi-process game generation pipeline
- **Training**: SGD-based learning from self-play data
- **Evaluation**: Stockfish integration for Elo estimation

## Features

### Chess Environment
- **119-plane board encoding** with 8-step position history
- **4672-action move space** covering all legal chess moves
- Efficient move encoding/decoding with `python-chess`

### Neural Network
- **ResNet architecture**: 15 residual blocks, 192 filters (configurable)
- **Dual heads**: Policy head (4672 actions) + Value head (position evaluation)
- **~10.8M parameters** for default configuration
- **Mixed precision (FP16)**: Training and inference support for 2-3x speedup on modern GPUs

### MCTS
- **PUCT algorithm** with exploration bonus
- **Dirichlet noise** at root for exploration
- **Temperature-based** action selection
- **Multi-backend support**: Pure Python, Cython, and C++ (pybind11)

### Training Pipeline
- **Multi-process self-play**: CPU actors + GPU learner
- **Replay buffer**: 1M position capacity with uniform sampling
- **SGD optimizer**: Learning rate scheduling, gradient clipping
- **Checkpointing**: Automatic model saving

### Evaluation
- **Arena**: Head-to-head matches between models
- **Stockfish integration**: Elo estimation at various skill levels
- **Self-play evaluation**: Track improvement over time

## Installation

### Prerequisites
- Python 3.13+
- CUDA-capable GPU (recommended for training)
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd alpha-zero-chess

# Install dependencies with uv
uv sync

# Verify installation
uv run pytest tests/ -v
```

## Quick Start

### Run Demos

Explore the implementation with educational demos:

```bash
# Neural network architecture demo
uv run python demo/network_demo.py

# MCTS step-by-step visualization
uv run python demo/mcts_demo.py

# Self-play game walkthrough
uv run python demo/selfplay_demo.py
```

### Train a Model

**Toy Example (Gaming Laptop - 8GB GPU, ~2 hours):**
```bash
# Quick training for testing (completes in ~2 hours on RTX 3060/4060)
uv run python scripts/train.py \
    --steps 5000 \
    --actors 8 \
    --filters 64 \
    --blocks 5 \
    --simulations 200 \
    --batch-size 2048 \
    --min-buffer 4096 \
    --batched-inference \
    --mcts-backend cython
```

**Small Network (Testing):**
```bash
# Basic training (small network for testing)
uv run python scripts/train.py \
    --steps 10000 \
    --actors 4 \
    --filters 64 \
    --blocks 5 \
    --simulations 400 \
    --batch-size 2048
```

**Full Training (Production - Requires 16GB+ GPU):**
```bash
# Production settings (requires powerful GPU)
uv run python scripts/train.py \
    --steps 100000 \
    --actors 16 \
    --filters 192 \
    --blocks 15 \
    --batch-size 4096 \
    --simulations 800 \
    --batched-inference \
    --mcts-backend cython
```

**Training parameters:**
- `--steps`: Number of training steps
- `--actors`: Number of self-play processes
- `--filters`: Network width (64, 128, 192, 256)
- `--blocks`: Network depth (5, 10, 15, 19)
- `--simulations`: MCTS simulations per move (default: 800)
- `--mcts-backend`: MCTS backend (`python`, `cython`, `cpp`)
- `--batched-inference`: Use centralized GPU inference server
- `--no-amp-training`: Disable mixed precision for training
- `--no-amp-inference`: Disable mixed precision for inference
- `--device`: `cuda` or `cpu`

### Play Against the Model

```bash
# Play interactively (architecture auto-detected from checkpoint name)
uv run python scripts/play.py \
    --checkpoint checkpoints/checkpoint_1000_f64_b5.pt \
    --color white \
    --simulations 400

# Or specify architecture manually for old checkpoints
uv run python scripts/play.py \
    --checkpoint checkpoints/checkpoint_1000.pt \
    --filters 64 \
    --blocks 5 \
    --color white \
    --simulations 400
```

**Commands during play:**
- Enter moves in UCI format (e.g., `e2e4`, `e7e8q`)
- Type `moves` to see legal moves
- Type `quit` to exit

### Evaluate Model Strength

**Quick Evaluation (Fast - 100 simulations per move):**
```bash
# Against random player (~2 minutes for 50 games)
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/checkpoint_1000_f64_b5.pt \
    --opponent random \
    --games 50 \
    --simulations 100

# Self-play evaluation
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/checkpoint_1000_f64_b5.pt \
    --opponent self \
    --games 20 \
    --simulations 100
```

**Full Evaluation (Slow - 800 simulations per move):**
```bash
# Against random player (~20 minutes for 100 games)
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/checkpoint_1000_f64_b5.pt \
    --opponent random \
    --games 100 \
    --simulations 800

# Against Stockfish (requires Stockfish installed)
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/checkpoint_1000_f64_b5.pt \
    --opponent stockfish \
    --stockfish-path "C:\path\to\stockfish.exe" \
    --stockfish-elo 1500 \
    --games 50 \
    --simulations 800
```

**Note:** Use fewer simulations (50-200) for quick evaluation, more simulations (400-800) for accurate strength measurement.

### Benchmark MCTS Performance

```bash
uv run python scripts/benchmark_mcts.py \
    --searches 100 \
    --simulations 800 \
    --backends python
```

## Architecture Details

### Board Encoding (119 planes, 8×8 each)

| Planes | Description |
|--------|-------------|
| 0-95 | Piece positions for 8 history steps (12 planes × 8 steps) |
| 96-99 | Castling rights (4 planes) |
| 100 | Side to move |
| 101-108 | Repetition counters |
| 109-118 | Move clocks (halfmove, fullmove) |

### Action Space (4672 actions)

| Range | Description |
|-------|-------------|
| 0-3583 | Queen-like moves (56 directions × 64 squares) |
| 3584-4095 | Knight moves (8 × 64 = 512) |
| 4096-4671 | Underpromotions (9 × 64 = 576) |

### Neural Network

```
Input (119, 8, 8)
    ↓
Conv 3×3, 192 filters → BatchNorm → ReLU
    ↓
ResidualBlock × 15
    ↓
    ├─→ Policy Head → (4672,)
    └─→ Value Head → (1,)
```

**Policy Head**: Conv 1×1 → BN → ReLU → FC(4672)\
**Value Head**: Conv 1×1 → BN → ReLU → FC(192) → ReLU → FC(1) → Tanh

### MCTS Algorithm

**PUCT Selection:**
```
a* = argmax_a [ Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a)) ]
```

**Dirichlet Noise (at root):**
```
P(s,a) = (1 - ε) · p_a + ε · η_a
where η ~ Dir(α), α = 0.3, ε = 0.25
```

**Temperature:**
- τ = 1.0 for first 30 moves (exploration)
- τ → 0 after move 30 (exploitation)

### Training

**Loss Function:**
```
L = L_policy + L_value
L_policy = -π^T · log(p)    # Cross-entropy
L_value = (z - v)²          # MSE
```

**Optimizer:** SGD with momentum
- Learning rate: 0.2 → 0.002 (drops at 100k, 300k, 500k steps)
- Momentum: 0.9
- Weight decay: 1e-4
- Gradient clipping: max_norm = 1.0

## Project Structure

```
alphazero/
├── chess_env/          # Chess game interface
│   ├── board.py        # GameState wrapper
│   ├── encoding.py     # Board encoding (119 planes)
│   └── moves.py        # Move encoding (4672 actions)
│
├── neural/             # Neural network
│   ├── blocks.py       # ResidualBlock, ConvBlock
│   ├── network.py      # AlphaZeroNetwork
│   └── loss.py         # Loss functions
│
├── mcts/               # Monte Carlo Tree Search
│   ├── base.py         # Abstract base classes
│   ├── evaluator.py    # Neural evaluator interface
│   └── python/         # Pure Python implementation
│       ├── node.py     # MCTSNode with PUCT
│       └── search.py   # MCTS search
│
├── training/           # Training pipeline
│   ├── trajectory.py   # Game trajectory storage
│   ├── replay_buffer.py # Replay buffer
│   └── learner.py      # Training loop
│
├── selfplay/           # Self-play generation
│   ├── game.py         # Single game execution
│   ├── actor.py        # Self-play actor
│   ├── batched_actor.py # Batched inference actor
│   ├── inference_server.py # GPU inference server
│   └── coordinator.py  # Multi-process orchestration
│
└── evaluation/         # Model evaluation
    ├── arena.py        # Match play
    └── stockfish.py    # Stockfish integration
```

## Testing

Run the test suite:

```bash
# All tests
uv run pytest tests/ -v

# Specific test modules
uv run pytest tests/test_chess_env.py -v
uv run pytest tests/test_neural.py -v
uv run pytest tests/test_mcts.py -v
uv run pytest tests/test_mcts_backends.py -v
uv run pytest tests/test_training.py -v
uv run pytest tests/test_batched_inference.py -v
uv run pytest tests/test_mixed_precision_inference.py -v
```

**Test coverage:**
- Chess environment: Move encoding, board encoding, game state
- Neural network: Architecture, forward pass, loss computation
- MCTS: Node operations, search algorithm, evaluators
- MCTS backends: Python, Cython, C++ implementation consistency and correctness
- Training: Trajectories, replay buffer, batch sampling
- Batched inference: InferenceServer, BatchedEvaluator, integration tests
- Mixed precision: FP16 inference, numerical stability, accuracy validation

## Hyperparameters

### Default Configuration (from AlphaZero paper)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Network filters | 192 | Can scale to 256 |
| Residual blocks | 15 | Can scale to 19 |
| MCTS simulations | 800 | Per move during self-play |
| c_puct | 1.25 | Exploration constant |
| Dirichlet α | 0.3 | Chess-specific (0.03 for Go) |
| Dirichlet ε | 0.25 | Noise weight at root |
| Temperature | 1.0 → 0 | τ=1 for moves 1-30, then greedy |
| Replay buffer | 1M positions | ~500k games |
| Batch size | 4096 | Training batch |
| Learning rate | 0.2 → 0.002 | Drops at 100k, 300k, 500k |
| Weight decay | 1e-4 | L2 regularization |

## Performance Notes

### Training Time Estimates

**Toy Example (8GB GPU - RTX 3060/4060):**
- Configuration: 5 blocks, 64 filters, 200 simulations, 8 actors
- Parameters: ~1M
- Training time: ~2 hours for 5000 steps
- Games generated: ~2000-3000 games
- Expected strength: Beats random play (80-90%), learns basic tactics
- **Perfect for:** Testing the pipeline, learning the codebase, quick experiments

**Small network (5 blocks, 64 filters):**
- ~1M parameters
- ~10-20 games/hour on CPU
- Suitable for testing and development

**Medium network (15 blocks, 192 filters):**
- ~10.8M parameters
- ~100-200 games/hour with 4 CPU actors + 1 GPU
- Reaches amateur level (~1500 Elo) after ~10k games

**Large network (19 blocks, 256 filters):**
- ~20M parameters
- Requires multi-GPU setup for reasonable training time
- Grandmaster level (~2500 Elo) requires ~44M games (AlphaZero paper)

### MCTS Backends

Three MCTS backends are available with different performance characteristics:

| Backend | Build Required | Simulations/sec | Notes |
|---------|---------------|-----------------|-------|
| Python  | No            | ~2,000          | Default, no compilation needed |
| Cython  | Yes           | ~2,400 (1.2x)   | Requires Cython build |
| C++     | Yes           | ~2,500 (1.2x)   | Requires pybind11 build |

> **Note:** The modest speedups (1.2x vs theoretical 5-50x) are because the bottleneck is in Python-side game state operations (`apply_action`, `get_observation`) and neural network evaluation, not the MCTS tree operations themselves.

#### Building Optimized Backends

```bash
# Build Cython backend
uv run python alphazero/mcts/cython/setup.py build_ext --inplace

# Build C++ backend
cd alphazero/mcts/cpp
uv run python setup.py build_ext --inplace
```

#### Choosing a Backend for Training

Use the `--mcts-backend` flag when training:

```bash
# Use Python backend (default, no build required)
uv run python scripts/train.py --mcts-backend python

# Use Cython backend (requires build)
uv run python scripts/train.py --mcts-backend cython

# Use C++ backend (requires build)
uv run python scripts/train.py --mcts-backend cpp
```

#### Using Backends Programmatically

```python
from alphazero import MCTSBackend
from alphazero.mcts import create_mcts, get_available_backends

# Check which backends are available
print(get_available_backends())  # [MCTSBackend.PYTHON, MCTSBackend.CYTHON, MCTSBackend.CPP]

# Create MCTS with specific backend
mcts = create_mcts(backend=MCTSBackend.CPP)

# Or use in self-play
from alphazero.selfplay import SelfPlayGame
game = SelfPlayGame(config, network, mcts_backend=MCTSBackend.CYTHON)
```

### Batched GPU Inference

For multi-actor training, use `--batched-inference` to enable centralized GPU inference:

```bash
# Recommended for 4+ actors with mixed precision
uv run python scripts/train.py \
    --actors 24 \
    --batched-inference \
    --mcts-backend cython
```

**Architecture:**
```
Actor 1 ──┐                              ┌── Response Queue 1
Actor 2 ──┼── Request Queue ── GPU ──────┼── Response Queue 2
Actor 3 ──┤    Server                    ├── Response Queue 3
Actor 4 ──┘                              └── Response Queue 4
```

**Benefits:**
- Batches inference requests from multiple actors for better GPU utilization
- Actors focus on MCTS tree operations (CPU) while GPU handles neural network
- Reduces memory usage (single GPU model instead of N CPU copies)
- Scales better with more actors
- **2-3x faster with mixed precision (FP16)** on modern GPUs

**When to use:**
- Standard mode (`--actors 1-2`): Each actor runs its own CPU inference
- Batched mode (`--actors 4+`): Centralized GPU inference server

### Mixed Precision Inference

Mixed precision (FP16) inference provides significant speedups on modern GPUs:

```bash
# Enable mixed precision (default on CUDA)
uv run python scripts/train.py --actors 8 --batched-inference

# Disable mixed precision inference
uv run python scripts/train.py --no-amp-inference

# Disable both training and inference mixed precision
uv run python scripts/train.py --no-amp-training --no-amp-inference
```

**Performance benefits:**
- **2-3x faster inference** on GPUs with Tensor Cores (Volta, Turing, Ampere, Ada)
- **50% less memory** usage for activations
- **Higher throughput** in batched inference mode
- **No accuracy loss** for AlphaZero inference
- **Numerically stable** with FP16-compatible masking (-1e4 instead of -inf)

**GPU compatibility:**
- ✅ NVIDIA RTX 20/30/40 series (Tensor Cores)
- ✅ NVIDIA V100, A100 (Tensor Cores)
- ⚠️ Older GPUs (limited benefit)
- ❌ CPU (automatically disabled)

**Implementation details:**
- Uses PyTorch's `torch.amp.autocast('cuda')` for automatic mixed precision
- FP16-safe masking with `-1e4` (within FP16 range of ±65,504)
- Automatic fallback to FP32 on CPU
- Compatible with both training and inference

See [docs/mixed_precision_inference.md](docs/mixed_precision_inference.md) for detailed documentation.

### Parallel MCTS with Virtual Loss

The implementation includes support for parallel MCTS simulations within a single actor using virtual loss:

```python
from alphazero.mcts.python.parallel import ParallelMCTS

# Run MCTS with multiple threads
mcts = ParallelMCTS(config)
policy, root, stats = mcts.search(
    state, evaluator,
    num_threads=4  # Parallel simulations
)
```

**Virtual Loss** temporarily penalizes nodes being explored by other threads, encouraging diverse exploration paths.

## Recent Updates

### v0.2.0 - Mixed Precision Inference (2026-01-29)
- ✅ **Mixed precision (FP16) inference** for 2-3x speedup on modern GPUs
- ✅ **Numerical stability fixes** for FP16 compatibility (using -1e4 instead of -inf)
- ✅ **Updated PyTorch API** to use `torch.amp.autocast('cuda')` (deprecated API removed)
- ✅ **Bug fixes** for game result handling (NoneType edge cases)
- ✅ **Comprehensive documentation** in `docs/mixed_precision_inference.md`
- ✅ **Test suite** for mixed precision validation

### v0.1.0 - Initial Implementation
- ✅ Complete AlphaZero implementation with ResNet architecture
- ✅ Multi-backend MCTS (Python, Cython, C++)
- ✅ Batched GPU inference server
- ✅ Multi-process self-play pipeline
- ✅ Replay buffer and SGD training

## Future Enhancements
Next Steps:
- [x] Batched GPU inference for actors
- [x] Virtual loss for parallel MCTS
- [x] Mixed precision inference optimization
- [ ] Opening book integration (only as an evaluation metric, DO NOT use to train network)
- [ ] Training visualization dashboard
- [ ] Web interface for playing against the model

Potential Steps:
- [ ] Endgame tablebase support
- [ ] Distributed training across multiple machines


## References

- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) (AlphaZero paper)
- [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) (AlphaGo Zero paper)
- [python-chess](https://python-chess.readthedocs.io/) library

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This implementation is based on the AlphaZero algorithm developed by DeepMind. The project structure and implementation details follow the methodology described in the AlphaZero paper.
