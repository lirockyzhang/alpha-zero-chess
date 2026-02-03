# AlphaZero Chess - Documentation

Complete documentation for the AlphaZero chess implementation with C++ backend acceleration.

## 🎉 Recent Updates

### 2026-02-02: Channel Update (119 → 122) ✅ VERIFIED
- **Upgraded to 122 input channels** for full AlphaZero specification compliance
- All 8 historical positions now fully encoded (T-8 complete: channels 109-121)
- **Verification Complete**: 10/10 tests passed (100% success rate)
- **Production Ready**: Confirmed shape `(*, 8, 8, 122)`, performance validated
- See: [Channel Update Guide](cpp-backend/channel-update-122.md) | [Testing Results](cpp-backend/testing-122-channels.md)
- **Breaking change**: Old 119-channel models incompatible, must retrain

## Quick Navigation

- **[Training & Evaluation Guide](training-guide.md)** - ⭐ **START HERE** - Complete guide with recommended parameters
- [Project Structure](../PROJECT_STRUCTURE.md) - Overall codebase organization
- [Implementation Plan](implementation-plan.md) - Ongoing C++ backend development
- [Test Suite](../alphazero-cpp/tests/) - Comprehensive test coverage
- [122-Channel Testing](cpp-backend/testing-122-channels.md) - Verification guide for new channel format

---

## Documentation Structure

### 📐 Architecture (`architecture/`)

Core algorithmic and system design documentation:

- **[AlphaZero Algorithm](architecture/alphazero-algorithm.md)** - Overview of the AlphaZero reinforcement learning algorithm
- **[Design Analysis](architecture/design-analysis.md)** - System architecture and design decisions
- **[Position History Encoding](architecture/position-history-encoding.md)** - ⭐ UPDATED: 8-position history input (122-channel tensor)

### ⚡ C++ Backend (`cpp-backend/`)

High-performance C++ implementation documentation:

- **[C++ Backend Guide](cpp-backend/cpp-backend-guide.md)** - Complete guide to using the C++ backend
- **[PyTorch Integration](cpp-backend/pytorch-integration.md)** - ⭐ Optimal C++ + PyTorch integration (3x performance boost)
- **[Component Overview](cpp-backend/component-overview.md)** - Comprehensive exploration of all C++ components
- **[Integration Report](cpp-backend/integration-report.md)** - Phase 4 completion report
- **[Channel Update (119→122)](cpp-backend/channel-update-122.md)** - ⭐ NEW: Complete change log for 122-channel upgrade
- **[122-Channel Testing](cpp-backend/testing-122-channels.md)** - ⭐ NEW: Verification and testing guide
- **[Python Encoder Note](cpp-backend/python-encoder-note.md)** - Why Python encoder wasn't updated
- **[Performance Benchmarks](cpp-backend/performance-benchmarks.md)** - Detailed performance metrics (6,192 sims/sec)
- **[Acceleration Technical Details](cpp-backend/acceleration-technical.md)** - Low-level optimization techniques

### 📚 Archive (`archive/`)

Historical documents and completed work (for reference only):

- **[Historical Progress](archive/historical-progress/)** - Phase 3 completion reports, bug fixes, reorganization summaries
- **[Old Plans](archive/old-plans/)** - Previous optimization proposals (quantization, mixed precision, etc.)

---

## Key Features Implemented

### ✅ Phase 1-3: Core C++ Backend (Complete)

1. **C++ MCTS Engine**
   - Batched leaf evaluation (AlphaZero specification)
   - Lock-free tree traversal
   - Zero-copy tensor interface
   - ~20-50x faster than Python MCTS

2. **C++ Self-Play Coordinator**
   - Multi-threaded game generation
   - Batched neural network inference
   - Lock-free queue for completed games
   - Temperature-based move selection

3. **Position History Encoding** ✅ VERIFIED
   - 8-position history in input tensor (channels 18-121) - **ALL 8 FULLY ENCODED**
   - Threefold repetition detection and enforcement
   - Repetition markers for each historical position
   - Full AlphaZero paper specification compliance
   - **Verified**: 10/10 comprehensive tests passing (2026-02-02)

4. **Zero-Copy Tensor Interface**
   - Direct buffer access from Python
   - NHWC memory layout (optimal for Tensor Cores)
   - 1.8x faster than copy-based encoding

5. **Performance Optimizations**
   - PyTorch channels_last (3x speedup)
   - OpenMP parallelization (2.2x speedup)
   - Memory pool optimization
   - Cache-line aligned structures

### 🚧 Phase 4-7: Full Backend Integration (In Progress)

See [Implementation Plan](implementation-plan.md) for details:

- C++ ReplayBuffer (lock-free circular buffer)
- C++ Trainer (asynchronous training loop)
- Unified training script (alphazero-cpp/scripts/train.py)
- Model compatibility verification
- Web interface testing

---

## Performance Metrics

### Current Performance (C++ Backend + PyTorch)

| Component | Metric | Value |
|-----------|--------|-------|
| MCTS Search | End-to-end | ~6,192 sims/sec (with NN) |
| MCTS Search | Peak | ~11,995 sims/sec |
| Chess Engine | Move generation | 189-422M nodes/sec |
| Position Encoding | Throughput | ~436K positions/sec (OpenMP) |
| Self-Play | Moves/sec | ~400-500 moves/sec |
| Game Generation | Latency | ~0.25s per 40-move game |

### Training Performance

| Configuration | Current | Target |
|---------------|---------|--------|
| Moves/sec during training | ~9 moves/sec | ~20-30 moves/sec |
| Training time (100 iter) | ~13-14 hours | ~5-7 hours |
| GPU Utilization | ~60-70% | ~85-95% |
| Games/iteration | 50-100 | 100-200 |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   AlphaZero Training Loop                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌─────────────────────┐      │
│  │  Self-Play (C++) │────────▶│ Replay Buffer       │      │
│  │  - MCTS Search   │         │ - Game storage      │      │
│  │  - Game playing  │         │ - Sampling          │      │
│  │  - 8-pos history │         └─────────────────────┘      │
│  └──────────────────┘                  │                    │
│          │                              │                    │
│          │ (observations)               │ (training samples) │
│          ▼                              ▼                    │
│  ┌──────────────────┐         ┌─────────────────────┐      │
│  │ Neural Network   │◀────────│  Trainer (Python)   │      │
│  │  (PyTorch GPU)   │         │  - SGD/Adam         │      │
│  │  - 192×15 ResNet │         │  - Batch updates    │      │
│  │  - channels_last │         │  - Checkpointing    │      │
│  └──────────────────┘         └─────────────────────┘      │
│          │                                                    │
│          │ (policy, value)                                   │
│          ▼                                                    │
│  ┌──────────────────┐                                       │
│  │   Evaluation     │                                       │
│  │   - Arena play   │                                       │
│  │   - ELO tracking │                                       │
│  │   - Stockfish    │                                       │
│  └──────────────────┘                                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Getting Started

### Training a Model

See **[Training & Evaluation Guide](training-guide.md)** for complete documentation.

```bash
# Quick test (~10 minutes)
uv run python alphazero-cpp/scripts/train.py \
    --iterations 2 --games-per-iter 5 --simulations 100 \
    --filters 64 --blocks 3

# Development (~2-4 hours, recommended starting point)
uv run python alphazero-cpp/scripts/train.py \
    --iterations 20 --games-per-iter 25 --simulations 400 \
    --filters 64 --blocks 5

# Production (~12-24 hours)
uv run python alphazero-cpp/scripts/train.py \
    --iterations 100 --games-per-iter 50 --simulations 800 \
    --filters 192 --blocks 15
```

### Running Tests

```bash
# Position history encoding tests (NEW!)
uv run python alphazero-cpp/tests/test_position_history.py

# MCTS correctness
uv run python alphazero-cpp/tests/test_python_bindings.py

# Performance benchmarks
uv run python alphazero-cpp/tests/benchmark_with_nn.py

# Cross-validation
uv run python alphazero-cpp/tests/test_cross_validation.py
```

### Evaluation

```bash
# Quick sanity check against random player
uv run python alphazero/scripts/evaluate.py \
    --checkpoint checkpoints/cpp_iter_20.pt \
    --opponent random --games 50 --simulations 200

# Endgame positions
uv run python alphazero/scripts/evaluate.py \
    --checkpoint checkpoints/cpp_iter_20.pt \
    --opponent endgame --simulations 400

# Against Stockfish (if installed)
uv run python alphazero/scripts/evaluate.py \
    --checkpoint checkpoints/cpp_iter_20.pt \
    --opponent stockfish --stockfish-elo 1500 --games 20
```

### Web Interface

```bash
# Play against your trained model
uv run python web/run.py --checkpoint checkpoints/cpp_iter_20.pt --simulations 400
# Open http://localhost:5000
```

---

## Technical Specifications

### Input Tensor Format

- **Shape**: `(batch, 8, 8, 122)` in NHWC layout **[UPDATED 2026-02-02]**
- **Channels 0-11**: Current position (12 piece planes: 6 types × 2 colors)
- **Channels 12-17**: Metadata
  - Channel 14: Color to move (always 1.0 from current player perspective)
  - Channel 15: Move count (normalized)
  - Channel 16: Castling rights
  - Channel 17: No-progress count (50-move rule)
- **Channels 18-121**: Position history (8 × 13 = 104 planes) **[ALL 8 POSITIONS FULLY ENCODED]**
  - T-1 (most recent): channels 18-30 (12 pieces + 1 repetition)
  - T-2: channels 31-43
  - T-3: channels 44-56
  - T-4: channels 57-69
  - T-5: channels 70-82
  - T-6: channels 83-95
  - T-7: channels 96-108
  - T-8: channels 109-121 (12 pieces + 1 repetition) **[NOW COMPLETE!]**

**Note**: Updated from 119 to 122 channels to fully support 8 historical positions as specified in AlphaZero paper.
See [Channel Update Documentation](cpp-backend/channel-update-122.md) for details.

### Network Architecture

- **Type**: ResNet with separate policy and value heads
- **Filters**: 192 (configurable: 64, 128, 192, 256)
- **Blocks**: 15 (configurable: 5, 10, 15, 20)
- **Parameters**: ~11M (for 192×15 configuration)
- **Memory Format**: `channels_last` for 3x performance
- **Input**: 119 planes (8×8×119)
- **Output**:
  - Policy: 4672 action logits (all possible moves)
  - Value: Single scalar (white win probability)

### MCTS Configuration

- **Simulations**: 800 (default, configurable)
- **PUCT constant**: 1.0
- **Dirichlet alpha**: 0.3 (for root exploration)
- **Exploration fraction**: 0.25
- **Temperature**: 1.0 (first 30 moves), 0.0 (thereafter)
- **Batch size**: 16-32 (leaf batching)

---

## Recent Updates

### 2026-02-02: Position History Encoding ✅ COMPLETE

- Implemented full 8-position history encoding (122 channels total)
- All 8 positions fully encoded (T-1 through T-8, channels 18-121)
- Threefold repetition detection and enforcement working
- **Verification**: 10/10 comprehensive tests passed (100% success rate)
- Performance: 388 moves/sec (no degradation, +2.5% memory)
- Documentation: [Position History Encoding](architecture/position-history-encoding.md)

### 2026-02-02: Documentation Reorganization

- Archived historical progress reports to `archive/historical-progress/`
- Archived old optimization plans to `archive/old-plans/`
- Created clean, focused documentation structure
- Updated README with current status and quick links

### 2026-02-02: Phase 2 Completion

- ✅ PyTorch channels_last optimization (3x performance boost)
- ✅ Optimal "input gate" pattern for C++ + PyTorch
- ✅ Project reorganization (backend-specific script folders)
- ✅ MCTS simplification (single batched implementation)
- ✅ Performance benchmarking with real trained models

---

## Contributing

When adding new features or documentation:

1. Update relevant documentation in `docs/`
2. Add tests to `alphazero-cpp/tests/`
3. Update performance benchmarks if applicable
4. Follow existing code style and documentation format
5. Update this README with links to new documents

---

## Additional Resources

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815) - Original DeepMind publication
- [chess-library](https://github.com/Disservin/chess-library) - C++ chess library used
- [PyTorch Documentation](https://pytorch.org/docs/) - Neural network framework
- [pybind11 Documentation](https://pybind11.readthedocs.io/) - C++/Python bindings

---

**Last Updated**: 2026-02-02

**Current Phase**: Position history encoding complete and verified (122 channels), proceeding with Phase 4-7 (full backend integration)

**Status**: ✅ **Production-ready** for self-play, inference, and training | 🚧 Full C++ training integration in progress
