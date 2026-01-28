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
- Mixed precision training (AMP) support

### MCTS
- **PUCT algorithm** with exploration bonus
- **Dirichlet noise** at root for exploration
- **Temperature-based** action selection
- **Multi-backend support**: Pure Python (implemented), Cython, C++ (stubs)

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

```bash
# Basic training (small network for testing)
uv run python scripts/train.py \
    --steps 10000 \
    --actors 4 \
    --filters 64 \
    --blocks 5

# Full training (production settings)
uv run python scripts/train.py \
    --steps 100000 \
    --actors 8 \
    --filters 192 \
    --blocks 15 \
    --batch-size 4096
```

**Training parameters:**
- `--steps`: Number of training steps
- `--actors`: Number of self-play processes
- `--filters`: Network width (64, 128, 192, 256)
- `--blocks`: Network depth (5, 10, 15, 19)
- `--simulations`: MCTS simulations per move (default: 800)
- `--device`: `cuda` or `cpu`

### Play Against the Model

```bash
uv run python scripts/play.py \
    --checkpoint checkpoints/checkpoint_10000.pt \
    --color white \
    --simulations 800
```

**Commands during play:**
- Enter moves in UCI format (e.g., `e2e4`, `e7e8q`)
- Type `moves` to see legal moves
- Type `quit` to exit

### Evaluate Model Strength

```bash
# Evaluate against random play
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/checkpoint_10000.pt \
    --opponent random \
    --games 100

# Evaluate against Stockfish
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/checkpoint_10000.pt \
    --opponent stockfish \
    --stockfish-path /path/to/stockfish \
    --stockfish-elo 1500 \
    --games 50
```

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
uv run pytest tests/test_training.py -v
```

**Test coverage:**
- Chess environment: Move encoding, board encoding, game state
- Neural network: Architecture, forward pass, loss computation
- MCTS: Node operations, search algorithm, evaluators
- Training: Trajectories, replay buffer, batch sampling

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

### MCTS Performance

**Pure Python backend:**
- ~50-100 simulations/second
- Suitable for development and testing

**Cython backend (not yet implemented):**
- Expected: ~500-1000 simulations/second (5-10x speedup)

**C++ backend (not yet implemented):**
- Expected: ~2000-5000 simulations/second (20-50x speedup)

## Future Enhancements
Next Steps:
- [ ] Cython MCTS implementation
- [ ] C++ MCTS implementation with pybind11
- [ ] Batched GPU inference for actors
- [ ] Opening book integration (only as an evaluation metrics, DO NOT use to train network)
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
