# Implementation Plan: Reorganize Project and Implement Full C++ Backend

## Overview

This plan addresses two major objectives:
1. **Reorganize project structure** - Move C++ backend out of alphazero/mcts/ and split scripts/ based on dependencies
2. **Implement complete C++ backend** - Reimplement all alphazero/ functionality in alphazero-cpp/ for maximum performance

**Key Principle:** Follow alphazero-cpp/ coding standards and architecture patterns throughout.

---

## Part 1: Project Reorganization

### 1.1 Remove C++ Backend from alphazero/mcts/

**Current Structure:**
```
alphazero/mcts/
├── cpp/
│   ├── __init__.py
│   └── backend.py  # Python wrapper for C++ MCTS
```

**Action:**
- Delete `alphazero/mcts/cpp/` directory entirely
- Update `alphazero/mcts/__init__.py` to remove C++ backend imports
- Update `create_mcts()` factory to only support Python and Cython backends
- The C++ MCTS will be accessed directly via `alphazero_cpp` module

**Files to Modify:**
- `alphazero/mcts/__init__.py` - Remove cpp imports and backend creation
- `alphazero/mcts/cpp/__init__.py` - DELETE
- `alphazero/mcts/cpp/backend.py` - DELETE

### 1.2 Create Scripts Folders

**Create:**
- `alphazero/scripts/` - For pure Python scripts
- `alphazero-cpp/scripts/` - For C++ backend scripts

### 1.3 Split and Move Scripts

**Move to `alphazero/scripts/`:**
- `scripts/train.py` → `alphazero/scripts/train.py`
- `scripts/evaluate.py` → `alphazero/scripts/evaluate.py`
- `scripts/play.py` → `alphazero/scripts/play.py`
- `scripts/benchmark_mcts.py` → `alphazero/scripts/benchmark_mcts.py`

**Move to `alphazero-cpp/scripts/`:**
- `scripts/cpptrain.py` → `alphazero-cpp/scripts/train.py` (rename for consistency)

**Delete:**
- `scripts/` directory (now empty)

**Update:**
- All moved scripts need path adjustments (sys.path.insert changes)
- README.md to reflect new script locations
- Any documentation referencing old paths

---

## Part 2: Complete C++ Backend Implementation

### 2.1 Architecture Overview

**Goal:** Make alphazero-cpp/ a complete, standalone, high-performance implementation that:
- Mirrors all functionality in alphazero/
- Uses C++ for performance-critical components
- Maintains model compatibility (same checkpoint format)
- Can be used interchangeably with Python backend

**Design Principles (following alphazero-cpp/ standard):**
1. **Self-contained implementations** - Define classes in scripts, not separate packages
2. **Direct C++ integration** - Use pybind11 bindings directly
3. **Performance-first** - Optimize for training and inference speed
4. **Minimal abstractions** - Simpler, more direct code paths
5. **Batched operations** - Leverage GPU efficiently

### 2.2 Components to Implement

**Note:** Configuration system skipped for now - will use simple parameter passing.

#### A. Neural Network (Keep PyTorch from Python)

**Current State:**
- Python: `alphazero/neural/network.py` - PyTorch implementation
- C++: None - currently uses PyTorch from Python

**Implementation Approach: Keep PyTorch (Python)**
- Continue using PyTorch from Python for neural network operations
- C++ components call Python for inference via pybind11
- Focus C++ optimization on MCTS, self-play, and data handling
- Simpler implementation, proven compatibility

**No C++ neural network implementation needed** - PyTorch stays in Python

**Python Bindings (for C++ to call Python NN):**
```python
# C++ will call Python evaluator
alphazero_cpp.set_evaluator(python_evaluator)
```

**Benefits:**
- No LibTorch dependency needed
- Full compatibility with existing checkpoints (no conversion)
- Simpler build process
- Still get major speedups from C++ MCTS and self-play
- Easier to maintain and debug

#### B. MCTS Simplification

**Current State:**
- `alphazero-cpp/include/mcts/search.hpp` - Single-threaded MCTS (root-only eval)
- `alphazero-cpp/include/mcts/batch_search.hpp` - Batched MCTS (proper AlphaZero)

**Simplification:**
- **DELETE** `search.hpp` and `search.cpp` (single-threaded, root-only version not needed)
- **RENAME** `batch_search.hpp` → `search.hpp`
- **RENAME** `batch_search.cpp` → `search.cpp`
- **RENAME** `BatchedMCTSSearch` class → `MCTSSearch`
- Update all references and bindings

**Rationale:**
- Only need the proper AlphaZero implementation (batched leaf evaluation)
- Simpler codebase with one MCTS implementation
- Clearer naming without "batched" prefix

#### C. Self-Play Engine

**Current State:**
- Python: `alphazero/selfplay/` - Multi-process actors with inference server
- C++: None

**Implementation:**
```
alphazero-cpp/
├── include/selfplay/
│   ├── game.hpp             # Single game execution
│   ├── actor.hpp            # Self-play actor
│   └── coordinator.hpp      # Multi-threaded coordinator (renamed from batch_coordinator)
├── src/selfplay/
│   ├── game.cpp
│   ├── actor.cpp
│   └── coordinator.cpp
```

**Features:**
- Multi-threaded self-play (C++ threads, not Python multiprocessing)
- Lock-free queues for game data
- Integrated with C++ MCTS and Python evaluator
- Temperature-based move selection
- Trajectory collection

**Python Bindings:**
```python
alphazero_cpp.SelfPlayGame(mcts, evaluator, config)
alphazero_cpp.SelfPlayActor(network, config)
alphazero_cpp.SelfPlayCoordinator(num_actors, network, config)  # Renamed from BatchedCoordinator
```

#### D. Training Engine

**Current State:**
- Python: `alphazero/training/` - PyTorch training loop
- C++: Partial - train.py uses PyTorch

**Implementation:**
```
alphazero-cpp/
├── include/training/
│   ├── replay_buffer.hpp    # Lock-free circular buffer
│   └── trainer.hpp          # Training loop coordinator
├── src/training/
│   ├── replay_buffer.cpp
│   └── trainer.cpp
```

**Features:**
- Lock-free replay buffer (C++ implementation)
- Batched sampling with zero-copy
- Integrated with PyTorch for gradient computation (via Python)
- Asynchronous training (train while generating games)
- Checkpoint management (via Python)
- **Replay buffer persistence** - Save/load buffer to disk for warm starts

**Replay Buffer Persistence:**
```python
# Save replay buffer to disk
alphazero_cpp.ReplayBuffer.save("replay_buffer/latest.npz")

# Load existing replay buffer on startup
buffer = alphazero_cpp.ReplayBuffer(max_size)
if os.path.exists("replay_buffer/latest.npz"):
    buffer.load("replay_buffer/latest.npz")
    print(f"Loaded {buffer.size()} samples from disk")
```

**Benefits:**
- Training can start immediately with existing data
- No need to wait for self-play games before first training step
- Faster iteration during development and debugging
- Can resume training with full replay buffer state

**Python Bindings:**
```python
alphazero_cpp.ReplayBuffer(max_size)
alphazero_cpp.ReplayBuffer.save(path)  # Save buffer to disk
alphazero_cpp.ReplayBuffer.load(path)  # Load buffer from disk
alphazero_cpp.Trainer(network, optimizer, config)
```

### 2.3 Evaluation Engine (Backend-Agnostic)

**IMPORTANT CHANGE:** Evaluation engine stays in the main `alphazero/` folder and works with both Python and C++ backends.

**Current State:**
- Python: `alphazero/evaluation/` - Arena, Stockfish integration
- Works with Python MCTS backends

**Enhancement (No C++ Implementation Needed):**
```
alphazero/evaluation/
├── __init__.py
├── arena.py              # Match playing (already exists)
├── stockfish.py          # Stockfish integration (already exists)
├── elo.py                # ELO calculation (already exists)
└── endgame_eval.py       # Endgame evaluation (already exists)
```

**Key Design:**
- Evaluation engine is **backend-agnostic**
- Works with any MCTS implementation (Python, Cython, or C++)
- Players are defined by their evaluator and MCTS config, not backend
- Both training scripts can use the same evaluation code

**Usage Example:**
```python
from alphazero.evaluation import Arena, MCTSPlayer
from alphazero.mcts import create_mcts
from alphazero.neural import AlphaZeroNetwork
import alphazero_cpp

# Python backend player
python_mcts = create_mcts(backend='python', config=mcts_config)
python_player = MCTSPlayer(python_mcts, python_evaluator)

# C++ backend player
cpp_mcts = alphazero_cpp.MCTSSearch(config)
cpp_player = MCTSPlayer(cpp_mcts, cpp_evaluator)

# Arena works with both
arena = Arena(python_player, cpp_player)
results = arena.play_games(num_games=100)
```

**No Changes Needed:**
- Existing evaluation code already works
- Just ensure C++ MCTS has same interface as Python MCTS
- Both backends produce compatible models

### 2.4 Unified Training Script

**Create:** `alphazero-cpp/scripts/train.py` (enhanced version of current cpptrain.py)

**Features:**
- Uses all C++ components (MCTS, inference, self-play, training)
- Multi-threaded self-play with C++ actors
- Asynchronous training pipeline
- Compatible checkpoint format with Python version
- Comprehensive performance metrics
- **Replay buffer persistence** - Load existing buffer on startup, save periodically

**Replay Buffer Integration:**
```python
# Create replay buffer
buffer = alphazero_cpp.ReplayBuffer(max_size=100000)

# Load existing buffer if available
buffer_path = "replay_buffer/latest.npz"
if os.path.exists(buffer_path):
    buffer.load(buffer_path)
    print(f"Loaded {buffer.size()} samples from previous runs")

# Training loop
for iteration in range(num_iterations):
    # Generate new self-play games
    games = self_play_coordinator.generate_games(num_games)
    buffer.add_games(games)

    # Train on buffer (includes old + new data)
    trainer.train(buffer, epochs=5)

    # Save buffer periodically
    if iteration % save_interval == 0:
        buffer.save(buffer_path)
        print(f"Saved replay buffer ({buffer.size()} samples)")
```

**Architecture:**
```
Main Thread
    ↓
├─→ C++ SelfPlayCoordinator
│   ├─→ C++ Actor 1 (thread)
│   ├─→ C++ Actor 2 (thread)
│   ├─→ C++ Actor 3 (thread)
│   └─→ C++ Actor 4 (thread)
│       ↓
│   C++ InferenceEngine (batched, GPU)
│       ↓
│   C++ ReplayBuffer (lock-free, persistent)
│
└─→ C++ Trainer (async)
    ↓
PyTorch Network (GPU)
```

### 2.5 Model Compatibility

**Critical Requirement:** Models must be interchangeable between Python and C++ backends.

**Checkpoint Format (Unified):**
```python
{
    'model_state_dict': {...},      # PyTorch state dict
    'optimizer_state_dict': {...},  # Optimizer state
    'iteration': int,
    'config': {
        'filters': int,
        'blocks': int,
        'simulations': int,
        ...
    },
    'backend': 'python' | 'cpp',    # Which backend created it
    'version': '1.0'
}
```

**Compatibility Requirements:**
1. Same network architecture (192×15 ResNet)
2. Same move encoding (4672 action space)
3. Same position encoding (119 planes)
4. Same checkpoint keys and structure
5. Both backends can load and continue training from either checkpoint

**Implementation:**
- C++ checkpoint loader uses libtorch to load PyTorch state dicts
- Python checkpoint loader works as-is
- Validation on load to ensure architecture matches

### 2.6 Web Interface Compatibility

**Requirement:** `web/run.py` must work with models from either backend without modification.

**Current State:**
- `web/run.py` imports from `alphazero` package
- Uses `alphazero.neural.AlphaZeroNetwork`
- Uses `alphazero.mcts.create_mcts`

**Solution:**
- Keep web interface using Python backend (alphazero package)
- Both backends produce compatible checkpoints
- Web interface loads checkpoint and creates Python MCTS + Network
- No changes needed to web/run.py

**Verification:**
1. Train model with `alphazero-cpp/scripts/train.py`
2. Save checkpoint as `checkpoints/cpp_model.pt`
3. Run `web/run.py --model checkpoints/cpp_model.pt`
4. Should work without errors

---

## Part 3: Implementation Order

### Phase 1: Reorganization (Low Risk)
1. ✅ Create `alphazero/scripts/` and `alphazero-cpp/scripts/`
2. ✅ Move scripts to new locations
3. ✅ Update import paths in moved scripts
4. Remove `alphazero/mcts/cpp/`
5. Update `alphazero/mcts/__init__.py`
6. Simplify C++ MCTS: Delete search.hpp/cpp, rename batch_search to search
7. Test that Python scripts still work

### Phase 2: C++ Infrastructure (Foundation)
1. Simplify MCTS naming (batch_search → search)
2. Update Python bindings for renamed MCTS
3. Test that existing C++ MCTS still works

### Phase 3: C++ Self-Play Engine
1. Implement SelfPlayGame
2. Implement SelfPlayActor (single-threaded first)
3. Implement multi-threaded SelfPlayCoordinator (renamed from batch_coordinator)
4. Add Python bindings
5. Test game generation with real trained model

### Phase 4: C++ Training Engine
1. Implement lock-free ReplayBuffer with save/load methods
2. Implement Trainer coordinator
3. Add Python bindings
4. Test training loop with real trained model
5. Test replay buffer persistence (save/load)

### Phase 5: Verification & Integration
1. Verify evaluation engine works with C++ backend (no changes needed)
2. Test model evaluation with real trained models
3. Create enhanced `alphazero-cpp/scripts/train.py` with buffer persistence
4. Test end-to-end training with replay buffer loading

### Phase 6: Final Verification
1. Train model with C++ backend (full training run, not toy example)
2. Train model with Python backend (full training run)
3. Verify checkpoints are compatible
4. Test web interface with both models
5. Compare training speed and model quality (real benchmarks with trained models)
6. Verify replay buffer persistence works correctly

---

## Part 4: File Structure (After Implementation)

```
alpha-zero-chess/
├── alphazero/                      # Pure Python package
│   ├── scripts/                    # NEW: Python scripts
│   │   ├── train.py               # Multi-process Python training
│   │   ├── evaluate.py            # Model evaluation
│   │   ├── play.py                # Interactive play
│   │   └── benchmark_mcts.py      # MCTS benchmarks
│   ├── chess_env/                 # Chess environment
│   ├── mcts/                      # MCTS (Python/Cython only)
│   │   ├── python/
│   │   ├── cython/
│   │   └── (cpp/ REMOVED)
│   ├── neural/                    # Neural network
│   ├── selfplay/                  # Self-play
│   ├── training/                  # Training
│   ├── evaluation/                # Evaluation (BACKEND-AGNOSTIC)
│   │   ├── arena.py              # Works with any backend
│   │   ├── stockfish.py          # Stockfish integration
│   │   ├── elo.py                # ELO calculation
│   │   └── endgame_eval.py       # Endgame evaluation
│   └── config.py                  # Configuration
│
├── alphazero-cpp/                 # Complete C++ backend
│   ├── scripts/                   # NEW: C++ scripts
│   │   └── train.py              # High-performance C++ training
│   ├── include/
│   │   ├── chess/                # Chess engine
│   │   ├── encoding/             # Position/move encoding
│   │   ├── mcts/                 # MCTS (simplified to one implementation)
│   │   ├── selfplay/             # NEW: Self-play engine
│   │   └── training/             # NEW: Training engine (with persistence)
│   ├── src/
│   │   ├── chess/
│   │   ├── encoding/
│   │   ├── mcts/
│   │   ├── selfplay/             # NEW
│   │   ├── training/             # NEW
│   │   └── bindings/
│   │       └── python_bindings.cpp  # All Python bindings
│   ├── tests/                    # C++ tests and benchmarks
│   └── CMakeLists.txt            # Build configuration
│
├── web/                           # Web interface (unchanged)
│   ├── app.py
│   └── run.py
│
├── checkpoints/                   # Shared checkpoints
├── replay_buffer/                 # NEW: Persistent replay buffer
│   └── latest.npz                # Latest buffer state
└── scripts/                       # DELETED (moved to subfolders)
```

---

## Part 5: Critical Files to Modify

### Reorganization Phase

**Delete:**
- `alphazero/mcts/cpp/__init__.py`
- `alphazero/mcts/cpp/backend.py`
- `scripts/` (entire directory after moving files)

**Modify:**
- `alphazero/mcts/__init__.py` - Remove C++ backend support
- `alphazero/scripts/train.py` - Update sys.path
- `alphazero/scripts/evaluate.py` - Update sys.path
- `alphazero/scripts/play.py` - Update sys.path
- `alphazero/scripts/benchmark_mcts.py` - Update sys.path
- `README.md` - Update script locations
- `PROJECT_STRUCTURE.md` - Update documentation

### C++ Implementation Phase

**Create (C++ Headers):**
- `alphazero-cpp/include/selfplay/game.hpp`
- `alphazero-cpp/include/selfplay/actor.hpp`
- `alphazero-cpp/include/selfplay/coordinator.hpp` (renamed from batch_coordinator)
- `alphazero-cpp/include/training/replay_buffer.hpp` (with save/load methods)
- `alphazero-cpp/include/training/trainer.hpp`

**Create (C++ Source):**
- Corresponding .cpp files for all headers

**Rename (C++ MCTS Simplification):**
- `alphazero-cpp/include/mcts/batch_search.hpp` → `search.hpp`
- `alphazero-cpp/src/mcts/batch_search.cpp` → `search.cpp`
- Class `BatchedMCTSSearch` → `MCTSSearch`

**Delete (C++ MCTS Simplification):**
- `alphazero-cpp/include/mcts/search.hpp` (old single-threaded version)
- `alphazero-cpp/src/mcts/search.cpp` (old single-threaded version)

**Modify:**
- `alphazero-cpp/CMakeLists.txt` - Update source file names, add new source files
- `alphazero-cpp/src/bindings/python_bindings.cpp` - Update class names, add all new bindings
- `alphazero-cpp/scripts/train.py` - Enhance with replay buffer persistence

**No Changes Needed (Evaluation):**
- `alphazero/evaluation/` - Already backend-agnostic, works as-is

---

## Part 6: Verification Plan

### 6.1 Reorganization Verification

**Test 1: Python Scripts Work**
```bash
cd alphazero/scripts
uv run python train.py --help
uv run python evaluate.py --help
uv run python play.py --help
uv run python benchmark_mcts.py --help
```
Expected: All scripts show help without import errors

**Test 2: C++ Script Works**
```bash
cd alphazero-cpp/scripts
uv run python train.py --help
```
Expected: Script shows help and finds alphazero_cpp

**Test 3: MCTS Factory**
```python
from alphazero.mcts import create_mcts, get_available_backends
backends = get_available_backends()
assert 'cpp' not in [b.value for b in backends]
```
Expected: C++ backend not in available backends

### 6.2 C++ Implementation Verification

**Test 1: Replay Buffer Persistence**
```python
import alphazero_cpp
import os

# Create and populate buffer
buffer = alphazero_cpp.ReplayBuffer(max_size=1000)
# ... add some games ...

# Save buffer
buffer.save("replay_buffer/test.npz")
assert os.path.exists("replay_buffer/test.npz")

# Load buffer in new instance
buffer2 = alphazero_cpp.ReplayBuffer(max_size=1000)
buffer2.load("replay_buffer/test.npz")
assert buffer2.size() == buffer.size()
```
Expected: Buffer saves and loads correctly

**Test 2: Checkpoint Compatibility**
```python
# Train with Python
python alphazero/scripts/train.py --iterations 1 --save-dir checkpoints/python

# Train with C++
python alphazero-cpp/scripts/train.py --iterations 1 --save-dir checkpoints/cpp

# Load Python checkpoint in C++
python alphazero-cpp/scripts/train.py --resume checkpoints/python/iter_1.pt

# Load C++ checkpoint in Python
python alphazero/scripts/train.py --resume checkpoints/cpp/cpp_iter_1.pt
```
Expected: Both can load and continue training from either checkpoint

**Test 3: Evaluation with C++ Backend**
```python
# Train with C++
python alphazero-cpp/scripts/train.py --iterations 10

# Evaluate using alphazero/evaluation (backend-agnostic)
python alphazero/scripts/evaluate.py \
    --model checkpoints/cpp_iter_10.pt \
    --backend cpp \
    --games 100
```
Expected: Evaluation works seamlessly with C++ backend

**Test 4: Web Interface**
```bash
# Train with C++
python alphazero-cpp/scripts/train.py --iterations 5

# Run web interface
python web/run.py --model checkpoints/cpp/cpp_iter_5.pt
```
Expected: Web interface loads and plays without errors

**Test 5: Warm Start Training**
```bash
# First training run
python alphazero-cpp/scripts/train.py --iterations 5
# Buffer saved to replay_buffer/latest.npz

# Second training run (warm start)
python alphazero-cpp/scripts/train.py --iterations 5
# Should load existing buffer and continue
```
Expected: Second run loads buffer and starts training immediately

**Test 6: Performance (Real Benchmarks with Trained Models)**
```bash
# First, train a model to use for benchmarking
python alphazero/scripts/train.py --iterations 10 --save-dir checkpoints/benchmark

# Benchmark Python backend training speed with REAL trained model
python alphazero/scripts/train.py \
    --resume checkpoints/benchmark/iter_10.pt \
    --iterations 1 \
    --games-per-iter 50
# Note: time per iteration, moves/sec, games/hour

# Benchmark C++ backend training speed with SAME trained model
python alphazero-cpp/scripts/train.py \
    --resume checkpoints/benchmark/iter_10.pt \
    --iterations 1 \
    --games-per-iter 50
# Note: time per iteration, moves/sec, games/hour

# Compare performance metrics
# Expected: C++ backend 2-5x faster
```

**Important:** All performance measurements must use real trained checkpoints, not random/untrained networks. This ensures realistic benchmarks that reflect actual training conditions.

---

## Part 7: Performance Targets

### Current Performance (Python + C++ MCTS)
- Self-play: ~9 moves/sec (800 sims, RTX 4060)
- Training: ~13-14 hours for 100 iterations

### Target Performance (Full C++ Backend)
- Self-play: ~20-30 moves/sec (2-3x faster)
- Training: ~5-7 hours for 100 iterations (2x faster)

### Optimization Strategies
1. **Multi-threaded self-play** - C++ threads instead of Python multiprocessing
2. **Lock-free data structures** - Replay buffer, game queues
3. **Zero-copy operations** - Direct memory access between C++ and PyTorch
4. **Batched inference** - Pre-allocated GPU buffers
5. **Asynchronous training** - Train while generating games
6. **Replay buffer persistence** - Start training immediately with existing data

---

## Part 8: Risk Mitigation

### Risk 1: Checkpoint Incompatibility
**Mitigation:**
- Implement checkpoint validation on load
- Add version field to checkpoints
- Test compatibility early and often
- Keep checkpoint format simple and well-documented

### Risk 2: Performance Regression
**Mitigation:**
- Benchmark each component individually
- Compare against Python baseline
- Profile to identify bottlenecks
- Iterate on optimization

### Risk 3: Complex C++ Implementation
**Mitigation:**
- Start with simple, working implementations
- Add optimizations incrementally
- Extensive testing at each phase
- Keep Python fallback available

### Risk 4: Web Interface Breaks
**Mitigation:**
- Don't modify web interface code
- Ensure checkpoint compatibility
- Test web interface after each phase
- Keep Python backend fully functional

### Risk 5: Replay Buffer Corruption
**Mitigation:**
- Add checksums to saved buffers
- Validate buffer on load
- Keep backup of previous buffer
- Graceful fallback to empty buffer if load fails

---

## Part 9: Success Criteria

### Reorganization Success
- ✅ All scripts moved to appropriate folders
- ✅ No import errors
- ✅ C++ backend removed from alphazero/mcts/
- ✅ Documentation updated

### Implementation Success
- ✅ Full C++ backend implemented
- ✅ Checkpoints are interchangeable
- ✅ Web interface works with both backends
- ✅ C++ backend is 2x+ faster
- ✅ Model quality is equivalent
- ✅ All tests pass
- ✅ Replay buffer persistence works correctly
- ✅ Evaluation engine works with both backends without modification

---

## Part 10: Timeline Estimate

**Phase 1 (Reorganization):** 1-2 hours
**Phase 2 (C++ Infrastructure - MCTS simplification):** 2-3 hours
**Phase 3 (C++ Self-Play):** 12-16 hours
**Phase 4 (C++ Training with Persistence):** 10-14 hours
**Phase 5 (Verification & Integration):** 4-6 hours
**Phase 6 (Final Verification):** 6-8 hours

**Total:** 35-49 hours of implementation work

**Note:** Significantly reduced from original estimate by:
- Skipping configuration system
- Keeping PyTorch in Python (no LibTorch needed)
- Simplifying MCTS (one implementation instead of two)
- Keeping evaluation engine in Python (backend-agnostic)
- More focused scope

---

## Notes

1. **Follow alphazero-cpp/ standard:** All new code follows the patterns in alphazero-cpp/ (self-contained, performance-first, minimal abstractions)

2. **Keep PyTorch in Python:** Neural network stays in Python for simplicity:
   - No LibTorch dependency needed
   - Full checkpoint compatibility (no conversion)
   - Simpler build process
   - Still get major speedups from C++ MCTS and self-play

3. **Simplify MCTS:** Only one MCTS implementation (proper AlphaZero with batched leaf evaluation):
   - Delete old single-threaded search.hpp/cpp
   - Rename batch_search → search
   - Clearer naming without "batched" prefix

4. **Skip configuration system:** Use simple parameter passing for now, can add later if needed

5. **Evaluation stays in Python:** Backend-agnostic design means no duplication needed:
   - Works with any MCTS backend
   - No C++ implementation required
   - Simpler maintenance

6. **Replay buffer persistence:** Major workflow improvement:
   - Training starts immediately with existing data
   - No cold start penalty
   - Better for iterative development
   - Saves buffer periodically during training

7. **Real benchmarks only:** All performance testing must use trained models, not toy examples:
   - Train a model first (10+ iterations)
   - Use that checkpoint for all benchmarks
   - Measure realistic training conditions

8. **Incremental approach:** Each phase is independently testable and adds value

9. **Backward compatibility:** Python backend remains fully functional throughout

10. **Model compatibility:** Critical requirement - extensive testing needed

11. **Performance focus:** C++ implementation prioritizes speed over flexibility

12. **Documentation:** Update all docs to reflect new structure and capabilities
